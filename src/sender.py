import dgl
import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import to_dgl
from scipy.sparse import csr_array
from scipy.sparse.csgraph import shortest_path
from itertools import combinations  
from collections import Counter
from src.utils import mtgnnConfig, accuracy
from src.processer import baseTrackConv


class SupContrastData(Dataset):
    def __init__(self, z, lable) -> None:
        self.z = z
        self.label = lable
        assert z.shape[0] == lable.shape[0]
    
    def __getitem__(self, index):
        return {
            'feat': self.z[index],
            'y': self.label[index]
        }
    
    def __len__(self):
        return self.z.shape[0]


class TreeTrackBuilder(nn.Module):
    # support raw input
    def __init__(self, config:mtgnnConfig):
        super(TreeTrackBuilder, self).__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.value_input = nn.Sequential(nn.Linear(config.feat_dim, config.n_heads*config.hidden), nn.Dropout(config.dr))
        self.query_input = nn.Sequential(nn.Linear(config.in_dim, config.n_heads*config.hidden))
        self.raw_query_input = nn.Sequential(nn.Linear(config.feat_dim, config.n_heads*config.hidden))
        self.key_input = nn.Sequential(nn.Linear(config.in_dim, config.n_heads*config.hidden))
        self.out_layer = nn.Sequential(nn.Linear(config.n_heads*config.hidden, config.hidden), nn.Dropout(config.dr), nn.ReLU())
    
    def sharp_alpha(self, alpha):
        entropy = -torch.sum(alpha*torch.log(alpha+1e-6), dim=-1)
        sharp_value = (math.e)**-entropy
        return sharp_value
    
    def attention(self, input_q, prototype, q_type='pre_emb'):
        if q_type == 'pre_emb':
            q = self.query_input(input_q)
        else:
            q = self.raw_query_input(input_q)
        q = torch.stack([q[:, i*self.config.hidden:(i+1)*self.config.hidden] for i in range(self.n_heads)])
        
        k = self.key_input(prototype)
        k = torch.stack([k[:, i*self.config.hidden:(i+1)*self.config.hidden] for i in range(self.n_heads)])
        
        alpha = torch.bmm(q, k.transpose(1, 2))
         
        return alpha

    def build(self, prototype, pre_emb, x, mid_mask):
        v = self.value_input(x)
        v = torch.stack([v[:, i*self.config.hidden:(i+1)*self.config.hidden] for i in range(self.n_heads)])
        
        alpha = torch.zeros((self.config.n_heads, v.shape[1], self.config.num_classes), device=v.device)
        
        no_mid_alpha = self.attention(x[~mid_mask], prototype)
        no_mid_alpha = F.softmax(no_mid_alpha, -1) # NOTE: pubmed
        
        alpha[:, ~mid_mask, :] = no_mid_alpha
        alpha[:, mid_mask, :] = 1/self.config.num_classes
        attn = torch.mean(alpha, dim=0)
        
        track_h = v.unsqueeze(-2)*alpha.unsqueeze(-1)
        track_h = track_h * self.sharp_alpha(alpha).unsqueeze(-1).unsqueeze(-1)
        track_h = torch.cat([track_h[i,:,:] for i in range(self.n_heads)], axis=-1)
        return track_h.transpose(0, 1), attn
    
    def forward(self, prototype, pre_emb, data):
        track_h, attn = self.build(prototype, pre_emb, data.x, data.mid_mask)
        return track_h, attn


class baseTrackBuilder(nn.Module):
    # support raw input
    def __init__(self, config:mtgnnConfig):
        super(baseTrackBuilder, self).__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.value_input = nn.Sequential(nn.Linear(config.feat_dim, config.n_heads*config.hidden), nn.Dropout(config.dr))
        self.query_input = nn.Sequential(nn.Linear(config.in_dim, config.n_heads*config.hidden))
        self.raw_query_input = nn.Sequential(nn.Linear(config.feat_dim, config.n_heads*config.hidden))
        self.key_input = nn.Sequential(nn.Linear(config.in_dim, config.n_heads*config.hidden))
        self.out_layer = nn.Sequential(nn.Linear(config.n_heads*config.hidden, config.hidden), nn.Dropout(config.dr), nn.ReLU())
    
    def sharp_alpha(self, alpha):
        entropy = -torch.sum(alpha*torch.log(alpha+1e-6), dim=-1)
        sharp_value = (math.e)**-entropy
        if torch.isnan(sharp_value).any():
            import ipdb; ipdb.set_trace()
        return sharp_value
    
    def attention(self, input_q, prototype, q_type='pre_emb'):
        if q_type == 'pre_emb':
            q = self.query_input(input_q)
        else:
            q = self.raw_query_input(input_q)
        q = torch.stack([q[:, i*self.config.hidden:(i+1)*self.config.hidden] for i in range(self.n_heads)])
        
        k = self.key_input(prototype)
        k = torch.stack([k[:, i*self.config.hidden:(i+1)*self.config.hidden] for i in range(self.n_heads)])
        
        alpha = torch.bmm(q, k.transpose(1, 2))
         
        return alpha

    def build(self, prototype, pre_emb, data):
        x = data.x
        v = self.value_input(x)
        v = torch.stack([v[:, i*self.config.hidden:(i+1)*self.config.hidden] for i in range(self.n_heads)])
        
        alpha1 = self.attention(pre_emb, prototype)
        # alpha2 = self.attention(x, prototype, 'raw_x')
        # alpha = F.softmax(
        #     (self.config.feat_weight * alpha1 + (1-self.config.feat_weight)*alpha2)/self.config.tau, dim=-1
        # )
        # hard_id = torch.load('/home/liyu/Graph_Neural_Network/cora_hard_id.bin')
        # alpha1[:, hard_id,:] += 10*F.one_hot(data.y[hard_id]).float()
        alpha = F.softmax(alpha1, -1) # NOTE: pubmed
        
        attn = torch.mean(alpha, dim=0)
        
        track_h = v.unsqueeze(-2)*alpha.unsqueeze(-1)
        track_h = track_h * self.sharp_alpha(alpha).unsqueeze(-1).unsqueeze(-1)
        track_h = torch.cat([track_h[i,:,:] for i in range(self.n_heads)], axis=-1)
        return track_h.transpose(0, 1), attn
    
    def forward(self, prototype, pre_emb, data):
        track_h, attn = self.build(prototype, pre_emb, data)
        return track_h, attn
    
        
class anchorTrackBuilder(nn.Module):
    def __init__(self, data, config):
        super(anchorTrackBuilder, self).__init__()
        self.conv_layer = baseTrackConv(feat_weight=1.0)
        self.layer_num = 4
        self.temperature = config.tau
        self.conv_h = self.init_conv_h(data)
        
        self.proj_head = nn.Sequential(nn.Linear(data.x.shape[1], config.hidden), nn.Dropout(config.dr)) # TODO:hidden dim
        self.fusion = nn.Linear(self.layer_num+1, 1, bias=False)
        
        self.optim = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-4)
        
    def init_conv_h(self, data):
        init_x = torch.zeros_like(data.x).to(data.x.device) # 不使用残差连接
        con_res = []
        for i in range(self.layer_num):
            if i == 0:
                conv_h = self.conv_layer(data.x, init_x, data.edge_index)
            else:
                conv_h = self.conv_layer(conv_h, init_x, data.edge_index)
            con_res.append(conv_h)
        return torch.stack(con_res, dim=1)
    
    @staticmethod
    def collect_fn(data_wrapper):
        x, y = [], []
        for xy_pair in data_wrapper:
            x.append(xy_pair['feat'])
            y.append(xy_pair['y'])
        return torch.stack(x), torch.stack(y)
    
    def get_geodesics_dist(self, sim, top_k):
        _, sim_id = torch.topk(sim, k=top_k)
        
        # id_list 2 id pair
        node_pair = []
        for a_node_sim in sim_id:
            node_pair.extend(list(combinations(a_node_sim.cpu().numpy(), 2)))
        node_pair = np.array(list(set(node_pair)))
        
        # build graph
        row, col = node_pair[:,0], node_pair[:,1]
        edge_weight = 1 - sim[row, col]
        edge_weight[edge_weight<0] = 0.01
        g = csr_array((edge_weight.cpu().numpy(), (row, col)), shape=(sim.shape[0], sim.shape[1]))      
        # caculate geo dist
        dist = shortest_path(csgraph=g, directed=False)
        return torch.from_numpy(dist).float().to(sim.device)
        
    def train_a_batch(self, x, label):
        # cross contrast in different massage-passing times
        label = label[:,0].unsqueeze(-1)
        z = self.proj_head(x)
        z = F.normalize(z, dim=-1, p=2).permute(1,0,2)
        
        indicator = torch.eq(label, label.T).float()
        logits_mask = torch.eye(indicator.shape[0], device=x.device)
        
        # compute logits
        logits = torch.matmul(z, z.transpose(2,1))/self.temperature
        
        # fusion each view logits
        sim = self.fusion(logits.permute(1,2,0)).squeeze(-1)
        
        # mask-out self-contrast cases
        sim = sim - logits_mask*1e9
        indicator = indicator - logits_mask
        
        # compute mean of likelihood over positive
        p = indicator / indicator.sum(1, keepdim=True).clamp(min=1.0)
        loss = F.cross_entropy(sim, p)
        return loss, sim
    
    @torch.no_grad()
    def inference(self, data, anchor, pred_anchor, is_vaild=False):
        # full batch inference
        x = torch.cat([data.x.unsqueeze(1), self.conv_h], dim=1)
        z = self.proj_head(x)
        z = F.normalize(z, dim=-1, p=2).permute(1,0,2)
        
        # caculate sim
        logits = torch.matmul(z, z.transpose(2,1))/self.temperature
        sim = self.fusion(logits.permute(1,2,0)).squeeze(-1)
        
        # vaild sim
        if is_vaild:
            p_sim = logits[0][anchor][:, ~anchor]
            non_anchor_num = torch.sum(~anchor)
            
            same_list, un_same_list = [], []
            for i in range(non_anchor_num):
                y_i = data.y[~anchor][i] 
                same_id, un_same_id = torch.where(data.y[anchor]==y_i)[0], torch.where(data.y[anchor]!=y_i)[0]
                same_sim, un_same_sim = p_sim[same_id, i].mean(), p_sim[un_same_id, i].mean()
                same_list.append(same_sim)
                un_same_list.append(un_same_sim)
            print(torch.mean(torch.stack(same_list)).item(), torch.mean(torch.stack(un_same_list)).item())
        
            # top 20 node in raw graph dist
            g = to_dgl(data)
            raw_dist = dgl.shortest_dist(g, return_paths=False)
            
            _, sim_id = torch.topk(sim, k=20)      
            print(raw_dist[:, sim_id[~anchor].view(-1)].float().mean())
        
        # get geo dist
        # dist = self.get_geodesics_dist(sim, 20)
        # dist = torch.where(torch.isinf(dist), 100, dist) # remove inf, max dist is 10
        
        # select anchor raw and non-anchor col
        # prototype_sim = sim[anchor][:, ~anchor]
        
        group_sim = []
        for c in range(data.y.max()+1):
            c_anchor_id = torch.where(pred_anchor==c)[0]
            group_sim.append(
                torch.topk(sim[anchor, :][c_anchor_id].T, 20)[0].mean(-1)) # select top 20 simlarity anchor
        _, predict_label = torch.stack(group_sim).max(dim=0)
        print(f'send acc: {accuracy(predict_label, data.y)}')
        return F.softmax(torch.stack(group_sim, dim=-1), dim=-1), z # return non anchor node track weight
    
    @torch.no_grad()
    def build(self, data, anchor, pred_anchor):
        track_w, z = self.inference(data, anchor, pred_anchor)
        track_h = z[0].unsqueeze(-1) * track_w.unsqueeze(1)
        return track_h.permute(2,0,1), track_w
    
    def prepare_dataset(self, anchor, data, pred_anchor):
        # add raw feat in z
        z = torch.cat([data.x.unsqueeze(1), self.conv_h], dim=1) # node num * view num * feat
        
        # init label
        label = pred_anchor.repeat(z.shape[1], 1).T # anchor num * view num
        
        # select anchor to training
        train_z = z[anchor] #  anchor num * view num * feat
        # train_label = label[anchor] # anchor num * view num
        
        return SupContrastData(train_z, label)
    
    def train_a_epoch(self, anchor, pred_anchor, data):
        dataset = self.prepare_dataset(anchor, data, pred_anchor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=self.collect_fn)
    
        for batch_x, batch_y in dataloader:
            batch_loss, batch_sim = self.train_a_batch(
                batch_x.to(data.x.device),
                batch_y.to(data.x.device)
            )
            
            self.optim.zero_grad()
            batch_loss.backward()
            self.optim.step()
        
        self.inference(data, anchor, pred_anchor, is_vaild=False)