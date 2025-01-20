import torch

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, APPNP
from pretrain.utils import *
from pretrain.gcnii import GCNIIdenseConv

class GnnLayer(nn.Module):
    def __init__(self, backbone, in_dim, out_dim, heads, K, alpha, layer_type='mid'):
        super(GnnLayer, self).__init__()
        if backbone == 'GCN':
            self.backbone = GCNConv(in_dim, out_dim)
        elif backbone == 'GAT':
            if layer_type == 'start':
                self.backbone = GATConv(in_dim, out_dim, heads)
            elif layer_type == 'mid':
                self.backbone = GATConv(in_dim*heads, out_dim, heads)
            elif layer_type == 'end':
                self.backbone = GATConv(in_dim*heads, out_dim)
            else:
                raise Exception('layer type is in [start, mid, end]')
        elif backbone == 'APPNP':
            self.backbone = APPNP(K, alpha, cached=True)
        else:
            raise Exception(f"The backbone '{backbone}' has not been realized")
    
    def forward(self, x, adj_t, x0=None):
        if x0 is None:
            x = self.backbone(x, adj_t)
        else:
            x = self.backbone(x, x0, adj_t)
        return x

class GCNII_model(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, nlayer):
        super(GCNII_model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(num_features, hidden_dim))
        for _ in range(nlayer):
            self.convs.append(GCNIIdenseConv(hidden_dim, hidden_dim))
        self.convs.append(torch.nn.Linear(hidden_dim, num_classes))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())

    def forward(self, x, adj_t, dropout):
        _hidden = []
        x = F.dropout(x, dropout ,training=self.training)
        x = F.relu(self.convs[0](x))
        _hidden.append(x)
        for i,con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, dropout ,training=self.training)
            beta = math.log(0.5/(i+1)+1)
            x = F.relu(con(x, adj_t, 0.1, _hidden[0],beta))
        x = F.dropout(x, dropout ,training=self.training)
        logit = self.convs[-1](x)
        return logit, x

class AppnpModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, nlayer, alpha, K):
        super(AppnpModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(num_features, hidden_dim))
        self.convs.append(APPNP(K, alpha))
        self.convs.append(torch.nn.Linear(hidden_dim, num_classes))
        
    def forward(self, x, adj_t, dropout):
        x = F.dropout(x, dropout ,training=self.training)
        x = F.relu(self.convs[0](x))
        for i,con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, dropout ,training=self.training)
            x = F.relu(con(x, adj_t))
        x = F.dropout(x, dropout ,training=self.training)
        logit = self.convs[-1](x)
        return logit, x

class Prior(nn.Module):
    def __init__(self, config:pretrainConfig):
        super(Prior, self).__init__()
        self.is_bns = config.is_bns
        self.convs = torch.nn.ModuleList()
        
        # start layer
        if config.backbone == 'GCNII':
            self.gcnii = GCNII_model(config.in_channels, config.hidden_channels, config.out_channels, config.num_layers)
        elif config.backbone == 'APPNP':
            self.appnp = AppnpModel(config.in_channels, config.hidden_channels, config.out_channels, config.num_layers, alpha=config.w, K=config.K)
        else:
            self.build_others(config)
                    
    def build_others(self, config):
        self.convs.append(
                GnnLayer(config.backbone, config.in_channels, config.hidden_channels, config.pheads, 
                        alpha=config.w, K=config.K, layer_type='start'))
        
        self.bns = torch.nn.ModuleList()
        if config.is_bns:
            self.bns.append(torch.nn.BatchNorm1d(config.hidden_channels))
        
        # mid layer
        for _ in range(config.num_layers - 2):
            self.convs.append(
                GnnLayer(config.backbone, config.hidden_channels, config.hidden_channels, config.pheads, 
                     alpha=config.w, K=config.K))
            if config.is_bns:
                self.bns.append(torch.nn.BatchNorm1d(config.hidden_channels))
        
        # last layer
        self.convs.append(
            GnnLayer(config.backbone, config.hidden_channels, config.out_channels, config.pheads, 
                     alpha=config.w, K=config.K, layer_type='end'))
    
    def forward(self, x: Tensor, adj_t, drop_rate: float = 0, backbone='GCN'):
        if backbone == 'GCNII':
            logit, x = self.gcnii(x, adj_t, drop_rate)
        elif backbone == 'APPNP':
            logit, x = self.appnp(x, adj_t, drop_rate)
        else:
            x0=None
            for i, conv in enumerate(self.convs[:-1]):
                x = F.dropout(
                    conv(x, adj_t, x0), drop_rate, self.training)
                if self.is_bns:
                    x = self.bns[i](x)
                x = F.relu(x)

            logit = F.dropout(
                self.convs[-1](x, adj_t, x0), drop_rate)
        return logit, x.detach()
        