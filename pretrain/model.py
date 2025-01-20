import torch

import torch.nn as nn
import torch.nn.functional as F
import pretrain.backbone as Bb

from torch import Tensor
from torch_geometric.typing import Adj
from pretrain.utils import *

class DropBlock:
    def __init__(self, dropping_method: str):
        super(DropBlock, self).__init__()
        self.dropping_method = dropping_method

    def drop(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        if self.dropping_method == 'Dropout':
            x = F.dropout(x, drop_rate)

        return x, edge_index


class GNNLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, backbone: str, heads: int = 1,
                 K: int = 1, w: float = 0, add_self_loops: bool = True, normalize: bool = True, bias: bool = True,
                 transform_first: bool = False):
        super(GNNLayer, self).__init__()
        self.dropping_method = 'Dropout'
        self.drop_block = DropBlock(self.dropping_method)
        self.transform_first = transform_first

        if backbone == 'GCN':
            self.backbone = Bb.BbGCN(add_self_loops, normalize)
        elif backbone == 'GAT':
            self.backbone = Bb.BbGAT(in_channels, heads, add_self_loops)
        elif backbone == 'APPNP':
            self.backbone = Bb.BbAPPNP(K, w, add_self_loops, normalize)
        else:
            raise Exception('The backbone has not been realized')

        # parameters
        self.weight = nn.Parameter(torch.Tensor(heads * in_channels, heads * out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.backbone.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0, last_layer=False):
        message_drop = 0

        if self.training:
            x, edge_index = self.drop_block.drop(x, edge_index, drop_rate)

        if self.transform_first:
            x = x.matmul(self.weight)

        out = self.backbone(x, edge_index, message_drop)

        if not self.transform_first:
            h = out.matmul(self.weight)
        if self.bias is not None:
            h += self.bias
        if last_layer:
            return h, out
        else:
            return h


class Prior(nn.Module):
    def __init__(self, config:pretrainConfig):
        super(Prior, self).__init__()
        self.is_bns = config.is_bns
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GNNLayer(config.in_channels, config.hidden_channels, config.backbone, w=config.w, K=config.K))
        self.bns = torch.nn.ModuleList()
        if config.is_bns:
            self.bns.append(torch.nn.BatchNorm1d(config.hidden_channels))
        for _ in range(config.num_layers - 2):
            self.convs.append(
                GNNLayer(config.hidden_channels, config.hidden_channels, config.backbone, w=config.w, K=config.K))
            if config.is_bns:
                self.bns.append(torch.nn.BatchNorm1d(config.hidden_channels))
        self.convs.append(
            GNNLayer(config.hidden_channels, config.out_channels, config.backbone, w=config.w, K=config.K))

    def decode(self, z, edge_label_index):
        # z所有节点的表示向量
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print(dst.size())   # (7284, 64)
        r = (src * dst).sum(dim=-1)
        # print(r.size())   (7284)
        return F.sigmoid(r)
    
    def forward(self, x: Tensor, edge_index: Adj, edge_label_index:Adj=None, drop_rate: float = 0, get_emb: bool=False):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, drop_rate)
            if self.is_bns:
                x = self.bns[i](x)
            x = F.relu(x)
        x, emb = self.convs[-1](x, edge_index, drop_rate, last_layer=True)
        
        if get_emb:
            return x, emb.detach()
        else:
            r = self.decode(emb, edge_label_index) # 链接预测任务
            return x, r

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()