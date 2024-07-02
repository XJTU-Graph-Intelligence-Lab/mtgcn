import os
import sys
data_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(data_root)

import torch

from src.utils import mtgnnConfig, prototype_init, EarlyStopping, accuracy, group_distance_ratio
from src.model import mtGNN

from pretrain.new_model import Prior
from pretrain.utils import pretrainConfig
from modis import Modis, Conf

def train_init(data, train_mask, pseudo_mask, stage, last_predict=None, args=None):
    pretrain_cfg = pretrainConfig(args.backbone, data, args.p_hidden, args.proir_layers)
    pretrain_model = Prior(pretrain_cfg).to(args.device)
    pretrain_model.load_state_dict(
        torch.load(data_root+f'/prior_checkpoint/{args.dataset}_{args.backbone}_{args.train_spilt}.pt', 
                   map_location=torch.device(args.device))
    )
    pretrain_model.eval()
    
    # 推理
    logit, node_embd = pretrain_model(data.x, data.adj_t, drop_rate=args.pdr, backbone=args.backbone)
    easy_node_pool = torch.load(data_root+f'/prior_checkpoint/{args.dataset}_{args.backbone}_{args.train_spilt}_easynode.pt')
    prototype, easy_node_pool = prototype_init(node_embd, data, train_mask, pseudo_mask, easy_node_pool, stage, last_predict)
    return prototype, easy_node_pool, node_embd


def train(data, label, train_mask, val_mask, test_mask, pseudo_mask, num_k, stage, last_predict=None, return_gr=False, args=None):
    if args is None:
        from args import args
    prototype, easy_node_pool, pre_emb = train_init(data, train_mask, pseudo_mask, stage, last_predict, args)
    # init model and optim
    print(f'mtgnn layer num:{args.layer_num}')
    cfg = mtgnnConfig(data, prototype, args.device, args)
    model = mtGNN(cfg).to(args.device)
    
    tb_optimizer = torch.optim.Adam(model.track_builder.parameters(), lr=args.tblr, weight_decay=args.tbwd)
    tp_optimizer = torch.optim.Adam([{"params":model.track_processer.parameters()}, 
                                        {"params":model.fc.parameters()}], lr=args.lr, weight_decay=args.tpwd) # 5e-4
    
    hard_node = torch.zeros_like(data.y, device='cuda:0')
    ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoint/MTGCN.pt')
    es = EarlyStopping(args.num_epochs_patience, path=ckpt_path)
    # end_epoch, start_epoch = args.end_epoch, args.start_epoch_p # 大数据集300 150, 小数据集30 100
    # mem = Modis(data.y, end_epoch-start_epoch, args.device)
    # mem_count = 0
    conf = Conf(data.y)
    max_val_acc = 0
    for e in range(500):
        if e % 2 != 0 or e < 20:
            val_loss_mt, logits = model.train_one_epoch(tb_optimizer, prototype, pre_emb, 
                                                data, label,easy_node_pool, train_mask,
                                                val_mask, pseudo_mask, e, 'tb', stage)
        else:
            val_loss_mt, logits = model.train_one_epoch(tp_optimizer, prototype, pre_emb, 
                                                data, label, easy_node_pool, train_mask,
                                                val_mask, pseudo_mask, e, 'tp', stage)
            
            # if e > args.start_epoch and mem_count < end_epoch-start_epoch:
            #     mem.update_memory(logits)
            #     mem_count += 1
               
        # get test/val acc
        val_acc_mt, test_acc_mt, logit, new_emb = model.eval_res(prototype, pre_emb, data, easy_node_pool, 
                        val_mask, test_mask, pseudo_mask, stage)
        if e > 150:
            hard_node += (torch.max(logit, -1)[1] != data.y)
        # early stop
        es(val_loss_mt, val_acc_mt, test_acc_mt, model)
        if es.early_stop:
            print(f"early stop, final mtgcn test acc is {es.best_test_acc}")
            break
        
        max_val_acc = max(val_acc_mt, max_val_acc)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=args.device))
    _, _, logit, new_emb = model.eval_res(prototype, pre_emb, data, easy_node_pool, 
                        val_mask, test_mask, pseudo_mask, stage)
    # _, hard_id = torch.topk(hard_node, 200)
    # torch.save(hard_id, f'{args.dataset}_hard_id.bin')
    # 统计本轮测试集分对的节点mask
    # predict = logit.argmax(1)
    # true_mask = predict.eq(data.y).cpu().numpy()
    # torch.save(true_mask, os.path.join(os.path.dirname(os.path.abspath(__file__)), f'mid_res/{args.dataset}_{stage}_trueNode.pt'))
    
    # if return_gr:
    #     model.over_smooth_analysis(prototype, pre_emb, data)
    #     return es.best_test_acc.item() 
    
    easy_node_pool, n_pseudo_mask = conf.get_easynode(logit, num_k, train_mask, pseudo_mask, True)
    easy_idx = easy_node_pool['idx']
    print(f"easy node acc: {accuracy(easy_node_pool['pred'][easy_idx], data.y[easy_idx])}")
    return n_pseudo_mask, easy_idx, easy_node_pool['pred'], es.best_test_acc.item(), max_val_acc  


if __name__ == '__main__':
    train()