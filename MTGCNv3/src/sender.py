import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
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
    
        