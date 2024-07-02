'''
Author: liyu1998 1293384713@qq.com
Date: 2024-01-20 15:48:04
LastEditors: liyu1998 1293384713@qq.com
LastEditTime: 2024-03-28 14:32:27
FilePath: /Graph_Neural_Network/MTGCNv3/mutil_stage_train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os
import sys
import torch
data_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(data_root)

from data.data_process import load_data
from args import args
from train import train
from train_prior import train_prior

def seed_torch(seed=1029):
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('seed init successful')

def update_pseudo(label, predict, easy_idx):
    label[easy_idx] = predict[easy_idx]
    return label

def mutil_stage_train():
    print(args)
    seed_torch()
    data = load_data(args.dataset, args.new_split, args.train_spilt, args.device)
    if len(data.train_mask.shape) > 1:
        train_mask, val_mask, test_mask = data.train_mask[:,0], data.val_mask[:,0], data.test_mask[:,0]
    else:
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    
    label = data.y.clone()
    num_k_dict = {
        'wisconsin': [5,0,0,0],
        'cora': [100, 50, 20, 1],
        'citeseer': [100, 50, 20, 1],
        'pubmed': [1000, 500, 300, 1], # 400,50,20,1
        'cs': [50, 10, 5, 1],
        'physics': [1500, 1500, 700, 10],
        'texas': [5,2,2,1],
        'cornell': [5,2,2,1]
    }
    num_k_list = num_k_dict[args.dataset]
    res, p_res = [], []
    pseudo_mask = torch.zeros_like(train_mask, dtype=bool, device=args.device)
    for s in range(4):   
        print(f'------------------stage {s} begin -------------------') 
        num_k = num_k_list[s]
        train_num = torch.sum(pseudo_mask).item()+0.0001
        label_acc = label[pseudo_mask].eq(data.y[pseudo_mask]).sum()/train_num
        print(f'train node num:{train_num}, pseudo node acc{label_acc.item()}')
        
        p_max_acc = train_prior(data, label, train_mask, val_mask, test_mask, pseudo_mask, num_k)

        if s == 0:    
            pseudo_mask, easy_idx, predict, max_acc = train(data, label, train_mask, val_mask, test_mask, pseudo_mask, num_k, stage=s)
        else:
            pseudo_mask, easy_idx, predict, max_acc = train(data, label, train_mask, val_mask, test_mask, pseudo_mask, num_k, s, predict)
        
        label = update_pseudo(label, predict, easy_idx)
        print(f'------------------stage {s} end -------------------') 
        res.append(max_acc)
        p_res.append(p_max_acc)
    print(p_res)
    print(res)


if __name__ == '__main__':
    mutil_stage_train()