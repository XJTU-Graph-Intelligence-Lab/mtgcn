import torch

import os.path as osp
import numpy as np
import torch.nn.functional as F

root_path = osp.dirname(osp.abspath(__file__))

class mtgnnConfig:
    def __init__(self, data, prior_emb, device, args=None):
        if args is None:
            from args import args
        self.args = args
        self.layer_num = args.layer_num
        self.n_heads = args.n_heads
        self.hidden = args.num_hidden
        self.dr = args.dr
        self.tau = args.tau
        self.fai1 = args.fai1
        self.feat_weight = args.a
        self.feat_dim, self.num_classes = self.get_input_dims_and_clsn(data)
        self.in_dim = self.get_prior_dims(prior_emb)
        self.struct_dict = self.get_struct_dict()
        self.device = device
    
    def get_input_dims_and_clsn(self, data):
        return data.x.shape[1], data.y.max().item() + 1
    
    def get_prior_dims(self, prior_emb):
        return prior_emb.shape[1]
    
    def get_struct_dict(self):
        struct_dict = {
            'builder': f'{self.args.ipt_conv}TrackBuilder',
            'processer': f'{self.args.ipt_conv}TrackProcesser'
        }
        return struct_dict
    
class anchorMtConfig(mtgnnConfig):
    def __init__(self, data, device, args=None):
        if args is None:
            from args import args
        self.layer_num = args.layer_num
        self.n_heads = 1
        self.hidden = args.num_hidden
        self.dr = args.dr
        self.tau = 0.5
        self.fai1 = args.fai1
        self.feat_weight = 0.9
        self.feat_dim, self.num_classes = self.get_input_dims_and_clsn(data)
        self.in_dim = self.hidden
        self.struct_dict = self.get_struct_dict()
        self.device = device


def accuracy(pred, labels):
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


@ torch.no_grad()
def prototype_init(h, data, train_mask, pseudo_mask, easy_node_pool, stage, last_predict):
    # easy node pool:当前阶段先验模型的伪标签
    # last_predict, pseudo_mask: 上一阶段的伪标签
    prototype_list = []
    idx_by_class = easy_node_pool['group']
    new_idx = []
    for i, c_idx in enumerate(idx_by_class):
        idx = torch.where((data.y==i) & (train_mask==True))[0] # add train set

        if stage == 0:
            idx = torch.cat([c_idx, idx]) 
        else:
            assert last_predict is not None
            l_idx = torch.where((last_predict==i) & (pseudo_mask==True))[0]  
            idx = torch.cat([l_idx, idx])      
            new_idx.append(idx)
        
        # NOTE: 异常情况
        if len(idx) == 0:
            num = int(h.shape[0]/(data.y.max().item()+1))
            idx = np.random.choice(np.arange(h.shape[0]), num, replace=False)
                
        prototype_list.append(
                torch.mean(h[idx], dim=0)
            )
    if stage > 0:  
        easy_node_pool['idx'] = torch.cat(new_idx)
        easy_node_pool['pred'] = last_predict
    return torch.stack(prototype_list), easy_node_pool


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_test_acc = -np.Inf
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_val_acc = None
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, val_acc, test_acc, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0
        
        if self.best_val_acc is None:
            self.best_val_acc = val_acc
            self.save_checkpoint(val_loss, model)
        else:
            # if self.best_val_acc < val_acc:
            #     self.best_val_acc = val_acc
            #     self.best_test_acc = test_acc
            #     self.save_checkpoint(val_loss, model)
            if self.best_test_acc <= test_acc:
                self.best_test_acc = test_acc
                self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
        
def group_distance_ratio(x, y, eps: float = 1e-5) -> float:
        num_classes = int(y.max()) + 1

        numerator = 0.
        for i in range(num_classes):
            mask = y == i
            dist = torch.cdist(x[mask].unsqueeze(0), x[~mask].unsqueeze(0))
            numerator += (1 / dist.numel()) * float(dist.sum())
        numerator *= 1 / (num_classes - 1)**2

        denominator = 0.
        for i in range(num_classes):
            mask = y == i
            dist = torch.cdist(x[mask].unsqueeze(0), x[mask].unsqueeze(0))
            denominator += (1 / dist.numel()) * float(dist.sum())
        denominator *= 1 / num_classes

        return numerator / (denominator + eps), numerator, denominator