import os
import sys
import optuna
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

def mutil_stage_train(args, trial):
    print(args)
    seed_torch()
    data = load_data(args.dataset, args.new_split, args.train_spilt, args.device)
    if len(data.train_mask.shape) > 1:
        train_mask, val_mask, test_mask = data.train_mask[:,2], data.val_mask[:,2], data.test_mask[:,2]
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
            pseudo_mask, easy_idx, predict, _, max_acc = train(data, label, train_mask, val_mask, test_mask, pseudo_mask, num_k, stage=s, args=args)
        else:
            pseudo_mask, easy_idx, predict, _, max_acc = train(data, label, train_mask, val_mask, test_mask, pseudo_mask, num_k, s, predict, args=args)
        
        label = update_pseudo(label, predict, easy_idx)
        print(f'------------------stage {s} end -------------------') 
        res.append(max_acc)
        p_res.append(p_max_acc)
    print(p_res)
    print(res)
    return max(res)


def set_search_space(trial):
    args.lr = trial.suggest_float("lr", 0.001, 0.05, log = True)
    args.tblr = trial.suggest_float("tblr", 0.001, 0.05, log = True)
    args.tbwd = trial.suggest_float("tbwd", 1e-6, 1e-4)
    args.tpwd = trial.suggest_float("tpwd", 1e-6, 1e-4)
    args.lw = trial.suggest_float("lw", 0.01, 0.7)
    args.layer_num = trial.suggest_categorical("layer_num", [64, 96, 132, 164, 196, 256])

    args.dr = trial.suggest_float("dropout", 0.1, 0.8)
    res = mutil_stage_train(args, trial)
    return res

if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(set_search_space, n_trials = 20)
    for item in study.trials:
        with open('result_'+str(args.dataset)+'_more_layer.txt', 'a') as f:
            f.write(f'{item}\n')