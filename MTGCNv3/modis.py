import torch

import torch.nn.functional as F

class Modis:
    def __init__(self, label, memory_size, device):
        self.num_class = label.max()+1
        self.memory_size = memory_size
        self.predict_memory = torch.zeros((label.shape[0], self.num_class), device=device)
    
    def update_memory(self, logits, memory_t=0.1):
        pre = F.softmax(logits, -1)
        pre = pre ** (1 / memory_t)
        pre = pre / pre.sum(dim=1, keepdim=True)
        
        self.predict_memory += pre
    
    def get_easynode(self, num_k, train_mask, pseudo_mask, add_pseudo=False):
        labeled_mask = torch.logical_or(train_mask, pseudo_mask)
        predict_memory = self.predict_memory / self.memory_size
        
        pre_log = torch.log(predict_memory)
        pre_log = torch.where(torch.isinf(pre_log), torch.full_like(pre_log, 0), pre_log)
        pre_entropy = torch.sum(torch.mul(-predict_memory, pre_log), dim=1)
        
        easy_node_idx = []
        pseudo_labels = torch.argmax(predict_memory, dim=1)
        for i in range(self.num_class):
            class_index = torch.where(
                torch.logical_and(pseudo_labels == i, ~labeled_mask))[0]
            sorted_index = torch.argsort(pre_entropy[class_index], dim=0, descending=False)
            if sorted_index.shape[0] >= num_k:
                sorted_index = sorted_index[:num_k]
            easy_node_idx.append(class_index[sorted_index])   
            if add_pseudo:
                pseudo_mask[class_index[sorted_index]] = True
             
        easy_node_pool = {
        'idx': torch.cat(easy_node_idx),
        'pred': pseudo_labels,
        'group': easy_node_idx
        }
        
        return easy_node_pool, pseudo_mask

class Conf:
    def __init__(self, label):
        self.label = label      
        self.num_class = label.max()+1
    
    @ torch.no_grad()
    def get_easynode(self, logits, num_k, train_mask, pseudo_mask, add_pseudo=False):
        labeled_mask = torch.logical_or(train_mask, pseudo_mask)
        
        dist = F.softmax(logits, -1)
        conf, pseudo = dist.max(-1)
        
        easy_node_idx = []
        for i in range(self.num_class):
            class_index = torch.where(
                torch.logical_and(pseudo == i, ~labeled_mask))[0]
            _, sorted_index = torch.sort(conf[class_index], dim=0, descending=True)
            if sorted_index.shape[0] >= num_k:
                sorted_index = sorted_index[:num_k]
            easy_node_idx.append(class_index[sorted_index])   
            if add_pseudo:
                pseudo_mask[class_index[sorted_index]] = True
        
        easy_node_pool = {
        'idx': torch.cat(easy_node_idx),
        'pred': pseudo,
        'group': easy_node_idx
        }
        
        return easy_node_pool, pseudo_mask
        
        