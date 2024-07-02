import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp

from args import args
from src.utils import mtgnnConfig, accuracy
from src.sender import *
from src.processer import *
            

class mtGNN(nn.Module):
    def __init__(self, config:mtgnnConfig) -> None:
        super().__init__()
        self.config = config
        self.track_builder = self.track_builder_trigger(config)
        self.track_processer = self.track_processer_trigger(config)
  
        self.fc = nn.Sequential(nn.Dropout(config.dr), 
                                nn.Linear(config.hidden, config.num_classes))
        self.loss_fn = nn.NLLLoss()
        self.bst_send_acc = 0
    
    @staticmethod
    def track_builder_trigger(config):
        builder = eval(config.struct_dict['builder'])
        return builder(config)
        
    @staticmethod
    def track_processer_trigger(config):
        processer = eval(config.struct_dict['processer'])
        return processer(config)
    
    def caculate_regular(self, attn, easy_node_pool):
        easy_node_idx, easy_node_pred, _ = easy_node_pool.values()
        attn_loss = F.cross_entropy(attn[easy_node_idx], easy_node_pred[easy_node_idx])
        return attn_loss
    
    def get_new_emb(self, prototype, pre_emb, data, is_training, stage):
        if stage == 100:
            if not osp.exists('0_attn.bin'):
                true_idx = np.random.choice(data.y.shape[0], int(data.y.shape[0]*0.6))
                rand_offset = np.random.randint(low=1, high=data.y.max().item(),size=(1,data.y.shape[0]))[0]
                rand_offset[true_idx] = 0
                pseudo_label = (data.y.cpu().numpy()+rand_offset)%(data.y.max().item()+1)
                pseudo_label = torch.tensor(pseudo_label, device='cuda:0')
                attn = F.one_hot(pseudo_label).float()
                torch.save(attn, '0_attn.bin')
            else:
                attn = torch.load('0_attn.bin')
            v = self.track_builder.value_input(data.x)
            track_h= (v.unsqueeze(1)*attn.unsqueeze(-1)).permute(1,0,2)
        else:
            track_h, attn = self.track_builder(prototype, pre_emb, data)
        new_emb = self.track_processer(track_h, attn, data, is_training)
        return new_emb, attn, track_h
    
    def forward(self, prototype, pre_emb, data, easy_node_pool, is_training, stage):
        new_emb, attn, track_h = self.get_new_emb(prototype, pre_emb, data, is_training, stage)
        # hard_idx = torch.load('/home/liyu/Graph_Neural_Network/texas_hard_id.bin')
        
        logit = self.fc(new_emb)
        # print(f'hard acc: {accuracy(logit[hard_idx].argmax(1), data.y[hard_idx]).item()}', end=' ')
        if self.track_builder.training or self.track_processer.training:
            send_acc = accuracy(attn.argmax(1), data.y).item()
            print(f'send acc:{send_acc:.4f}', end=' ')
            if send_acc > self.bst_send_acc:
                self.bst_send_acc = send_acc
                # torch.save(attn, f'attn_res/{args.dataset}_send_acc.pt')
            attn_loss = self.caculate_regular(attn, easy_node_pool)
            print(f'attn loss:{attn_loss:.4f}', end=' ')
            return F.log_softmax(logit, dim=1), attn_loss
        else:
            return F.log_softmax(logit, dim=1), new_emb
        
    def train_one_epoch(self, optimizer, prototype, pre_emb, data, label, 
                        easy_node_pool, train_mask, val_mask, pseudo_mask, e, train_part, stage):
        if train_part == 'tb': 
            self.track_builder.train()
            self.track_processer.eval()
        else:
            self.track_builder.eval()
            self.track_processer.train()
        
        optimizer.zero_grad()
        if self.track_builder.training or self.track_processer.training:
            is_training = True
        else:
            is_training = False
        logit, attn_loss = self.forward(prototype, pre_emb, data, easy_node_pool, is_training, stage)
        
        train_loss = self.loss_fn(logit[train_mask], label[train_mask]) 
        pseudo_loss = self.loss_fn(logit[pseudo_mask], label[pseudo_mask])
        
        if torch.isnan(pseudo_loss):
            all_loss = train_loss + self.config.fai1*attn_loss
        else:
            all_loss = train_loss + args.lw*pseudo_loss + self.config.fai1*attn_loss
        val_loss_mt = self.loss_fn(logit[val_mask], label[val_mask]) + self.config.fai1*attn_loss
        
        all_loss.backward()
        optimizer.step()
        
        train_acc = accuracy(logit.argmax(1)[train_mask], data.y[train_mask])
        # print(f'prototype dist {F.pdist(prototype, 2)}', end=' ')
        print(f'mtGNN epoch {e} loss {all_loss:.4f}, train acc {train_acc:.4f}', end=' ')
        return val_loss_mt, logit
    
    @torch.no_grad()
    def eval_res(self, prototype, pre_emb, data, easy_node_pool, val_mask, test_mask, pseudo_mask, stage):
        self.eval()
        logit, new_emb = self.forward(prototype, pre_emb, data, easy_node_pool, False, stage)
        
        val_acc_mt = accuracy(logit.argmax(1)[val_mask], data.y[val_mask])
        test_acc_mt = accuracy(logit.argmax(1)[test_mask], data.y[test_mask])
        pseudo_acc = accuracy(logit.argmax(1)[pseudo_mask], data.y[pseudo_mask])
        print(f'val acc {val_acc_mt:.4f}, test acc {test_acc_mt:.4f}, pseudo acc {pseudo_acc:.4f}')
        return val_acc_mt, test_acc_mt, logit, new_emb
    
    @ torch.no_grad()
    def over_smooth_analysis(self, prototype, pre_emb, data):
        track_h, attn = self.track_builder(prototype, pre_emb, data)
        self.track_processer.get_group_ratio(track_h, attn, data, args.dataset)
        

class anchorMtGNN(nn.Module):
    def __init__(self, track_builder, track_processer, config):
        super().__init__()
        self.track_builder = track_builder
        self.track_processer = track_processer
        self.classify = nn.Sequential(nn.Dropout(config.dr), nn.Linear(config.hidden, config.num_classes))
        
        self.tp_optim = torch.optim.Adam([{"params":self.track_processer.parameters()}, 
                                         {"params":self.classify.parameters()},
                                        ], lr=0.005, weight_decay=1e-4)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def get_new_emb(self, data, anchor, pred_anchor):
        track_h, track_w = self.track_builder.build(data, anchor, pred_anchor)
        new_emb = self.track_processer(track_h, track_w, data)
        return new_emb
    
    def forward(self, data, anchor, pred_anchor):
        track_h, track_w = self.track_builder.build(data, anchor, pred_anchor)
        new_emb = self.track_processer(track_h, track_w, data)
        logit = self.classify(new_emb)
        
        return logit, F.cross_entropy(track_w[anchor], pred_anchor)
    
    def train_one_epoch(self, anchor, pred_anchor, data, train_mask, val_mask, e, train_part):
        if train_part == 'tb':
            self.track_builder.train()
            # TODO: 两部分应该由一个总的loss去进行更新
            self.track_builder.train_a_epoch(anchor, pred_anchor, data)
            self.track_processer.eval()
        else:
            self.track_builder.eval()
            self.track_processer.train()
            
            self.tp_optim.zero_grad()
            logit, w_loss = self.forward(data, anchor, pred_anchor)
            
            train_loss = self.loss_fn(logit[train_mask], data.y[train_mask]) + 0.*w_loss
            val_loss_mt = self.loss_fn(logit[val_mask], data.y[val_mask]) + 0.1*w_loss
            train_loss.backward()
            self.tp_optim.step()
        
            train_acc = accuracy(logit.argmax(1)[train_mask], data.y[train_mask])
            # print(f'prototype dist {F.pdist(prototype, 2)}', end=' ')
            print(f'mtGNN epoch {e} loss {train_loss:.4f}, train acc {train_acc:.4f}', end=' ')
            return val_loss_mt
    
    @ torch.no_grad()
    def eval_res(self, data, anchor, pred_anchor, val_mask, test_mask):
        self.track_builder.eval()
        self.track_processer.eval()
        
        logit, _ = self.forward(data, anchor, pred_anchor)
        val_acc_mt = accuracy(logit.argmax(1)[val_mask], data.y[val_mask])
        test_acc_mt = accuracy(logit.argmax(1)[test_mask], data.y[test_mask])
        print(f'val acc {val_acc_mt:.4f}, test acc {test_acc_mt:.4f}')
        return val_acc_mt, test_acc_mt, logit
        
    

class stageLinker:
    # 两个阶段间的迭代处理
    def __init__(self, n_config:mtgnnConfig, p_emb_weight, esy_th): 
        self.p_emb_weight = p_emb_weight
        self.esy = esy_th
        self.num_classes = n_config.num_classes
    
    def update_emb(self, p_emb, node_emb):
        return self.p_emb_weight*p_emb + (1-self.p_emb_weight)*node_emb
    
    def get_prototype(self, merge_cfd, merge_pred, updated_emb, data):
        easy_node_idx = merge_cfd>self.esy
        prototype_list = []
        for i in range(self.num_classes):
            c_idx = torch.where((merge_pred==i) & (merge_cfd>self.esy))[0]
            if len(c_idx) > 0:
                prototype_list.append(
                    torch.mean(updated_emb[c_idx], dim=0)
                )
            else:
                num = int(updated_emb.shape[0]/(self.num_classes))
                idx = np.random.choice(np.arange(updated_emb.shape[0]), num, replace=False)
                prototype_list.append(
                    torch.mean(updated_emb[idx], dim=0)
                )
        easy_node_pool = {
            'idx': easy_node_idx,
            'pred': merge_pred[easy_node_idx]
        }
        
        print(f'easy node acc {accuracy(merge_pred[easy_node_idx], data.y[easy_node_idx]).item():.4f}, easy node num {easy_node_idx.sum()}')
        # anlysis_predict(predit, data.y, easy_node_idx, confidence)
        return torch.stack(prototype_list), easy_node_pool
    
    @staticmethod
    def update_cfd(logit, p_logit):
        cfd, predit = torch.max(logit, dim=-1)
        p_cfd, p_pred = torch.max(p_logit, dim=1)
        
        idx = torch.where(p_cfd<cfd)
        # curr conf > prior conf
        p_cfd[idx] = cfd[idx]
        p_pred[idx] = predit[idx]
        return p_cfd, p_pred
    
    @torch.no_grad()
    def link_stage(self, p_logit, p_emb, data, logit, node_embd):
        """
        连接两个stage 迭代cfd、node emb、prototype
        Args:
            p_cfd (_type_): 上一阶段的置信度
            p_emb (_type_): 上一阶段的节点表示
            data (_type_): 图
            logit: 现阶段logit
            node_embd: 现阶段节点emb
        """
        logit = math.e ** logit
        merge_cfd, merge_pred = self.update_cfd(logit, p_logit)
        merge_emb = self.update_emb(p_emb, node_embd)
        new_prototype, n_easynode_pool = self.get_prototype(merge_cfd, merge_pred, merge_emb, data)
        return merge_emb, new_prototype, n_easynode_pool