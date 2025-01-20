import torch

import torch.nn as nn
import torch.nn.functional as F

from src.utils import mtgnnConfig, group_distance_ratio

from typing import Optional, Tuple
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from einops import rearrange, repeat

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from args import args


class baseTrackConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, feat_weight: float,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.feat_weight = feat_weight

        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        support = self.feat_weight*x
        initial = (1-self.feat_weight)*x_0
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=support, edge_weight=edge_weight,
                            size=None)+initial
        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class TreeTrackProcesser(nn.Module):
    def __init__(self, config):
        super(TreeTrackProcesser, self).__init__()  
        self.n_classes = config.num_classes
        self.n_layer = config.layer_num
        self.n_heads = config.n_heads
        self.dr = config.dr
        self.input_layer = nn.Sequential(nn.Dropout(config.dr), 
                                        nn.Linear(config.hidden, config.hidden*config.n_heads),
                                        nn.ReLU()) 
        self.track_convs = nn.ModuleList()
        for _ in range(config.layer_num):
            self.track_convs.append(baseTrackConv(config.feat_weight))

        self.post_process = nn.Dropout(config.dr)
        self.extra_embeding = nn.Linear(config.n_heads, 1)
        
    def message_passing(self, track_h, edge_index, x0, is_training):
        track_h =  rearrange(track_h, 'c n h -> n (c h)')
        init_x = repeat(self.input_layer(x0), 'n h -> n (c h)', c=self.n_classes)
        
        for i,con in enumerate(self.track_convs):
            if i == 0:
                tracks = F.dropout(track_h, self.dr, training=is_training)
                tracks = con(tracks, init_x, edge_index)
            else:
                tracks = F.dropout(tracks, self.dr, training=is_training)
                tracks = con(tracks, init_x, edge_index)
        
        tracks = rearrange(tracks, 'b (c h) -> c b h', c=self.n_classes)
        return tracks  
    
    def get_message(self, attn, track_message):
        new_emb = torch.sum(attn.T.unsqueeze(-1)*track_message, dim=0).squeeze()
        
        new_emb = self.post_process(new_emb).squeeze()
        if self.n_heads != 1: 
            new_emb = rearrange(new_emb, 'b (e h) -> b h e', e=self.n_heads)

            new_emb = self.extra_embeding(new_emb).squeeze()
        return new_emb  
    
    def forward(self, track_h, attn, data, is_training):
        tracks = self.message_passing(track_h, data.adj_t, data.x, is_training)
        new_emb = self.get_message(attn, tracks) 
        return new_emb   

class baseTrackProcesser(nn.Module):
    def __init__(self, config:mtgnnConfig):
        super(baseTrackProcesser, self).__init__()
        self.n_classes = config.num_classes
        self.n_layer = config.layer_num
        self.n_heads = config.n_heads
        self.dr = config.dr
        
        self.input_layer = nn.Sequential(nn.Dropout(config.dr), 
                                        nn.Linear(config.feat_dim, config.hidden*config.n_heads),
                                        nn.ReLU()) 
        
        self.track_convs = nn.ModuleList()
        for _ in range(config.layer_num):
            self.track_convs.append(baseTrackConv(config.feat_weight))

        self.post_process = nn.Dropout(config.dr)
        self.extra_embeding = nn.Linear(config.n_heads, 1)

    def message_passing(self, track_h, edge_index, x, is_training):
        track_h =  rearrange(track_h, 'c n h -> n (c h)')
        init_x = self.input_layer(x)
        init_x = repeat(init_x, 'n h -> n (c h)', c=self.n_classes)
        
        for i,con in enumerate(self.track_convs):
            if i == 0:
                tracks = F.dropout(track_h, self.dr, training=is_training)
                tracks = con(tracks, init_x, edge_index)
            else:
                tracks = F.dropout(tracks, self.dr, training=is_training)
                tracks = con(tracks, init_x, edge_index)
        
        tracks = rearrange(tracks, 'b (c h) -> c b h', c=self.n_classes)
        return tracks   
    
    def get_message(self, attn, track_message):
        new_emb = torch.sum(attn.T.unsqueeze(-1)*track_message, dim=0).squeeze()
        
        new_emb = self.post_process(new_emb).squeeze()
        if self.n_heads != 1: 
            new_emb = rearrange(new_emb, 'b (e h) -> b h e', e=self.n_heads)

            new_emb = self.extra_embeding(new_emb).squeeze()
        return new_emb  
    
    def forward(self, track_h, attn, data, is_training):
        tracks = self.message_passing(track_h, data.adj_t, data.x, is_training)
        new_emb = self.get_message(attn, tracks) 
        return new_emb 
    
    @ torch.no_grad()
    def get_group_ratio(self, track_h, attn, data, data_name):
        import pandas as pd
        res = []
        def get_message_in_track(tracks, attn):
            tracks = rearrange(tracks, 'b (c h) -> c b h', c=self.n_classes)
            new_emb = torch.sum(attn.T.unsqueeze(-1)*tracks, dim=0).squeeze()
            if self.n_heads != 1: 
                new_emb = rearrange(new_emb, 'b (e h) -> b h e', e=self.n_heads)
                new_emb = self.extra_embeding(new_emb).squeeze()
            return new_emb
            
        edge_index = data.adj_t
        
        track_h =  rearrange(track_h, 'c n h -> n (c h)')
        init_x = self.input_layer(data.x)
        init_x = repeat(init_x, 'n h -> n (c h)', c=self.n_classes)
        
        for i,con in enumerate(self.track_convs):
            if i == 0:
                tracks = con(track_h, init_x, edge_index)
            else:
                tracks = con(tracks, init_x, edge_index)
            gr_info = group_distance_ratio(
                get_message_in_track(tracks, attn)[data.test_mask], data.y[data.test_mask]
            )
            res.append(
                {'layer_num': i, 'group_ratio': gr_info[0], 'ex_class': gr_info[1], 'in_class': gr_info[2]}
            )
        df = pd.DataFrame(res)
        df.to_csv(f'out_class_anlysis_{data_name}.csv')
        
    
    
class diffTrackProcesser(baseTrackProcesser):
    def __init__(self, config: mtgnnConfig):
        super(diffTrackProcesser, self).__init__(config)
        self.track_convs = nn.ModuleList()
        for _ in range(64):
            self.track_convs.append(baseTrackConv(config.feat_weight))
        
        self.merge_group = nn.Sequential(
            nn.Linear(config.hidden*3, config.hidden),
            nn.Dropout(config.dr)
        )
    
    def forward(self, track_list, attn, data, pre_emb):
        track_res = []
        # 计算两组轨道的值
        for i in range(2):
            track_h = track_list[i]
            tracks = self.message_passing(track_h, data.edge_index, data.x)
            track_res.append(self.get_message(attn, tracks)) 
        merge_input = torch.cat([
            track_res[0], track_res[1], track_res[0]-track_res[1]
            ], dim=-1)   
        new_emb = 0.5*self.merge_group(merge_input) + 0.5*track_res[0]  
        return new_emb      