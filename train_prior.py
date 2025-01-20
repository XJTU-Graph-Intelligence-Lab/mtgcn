import os
import sys
data_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(data_root)

import torch

import torch.nn.functional as F

from data.data_process import load_data
from modis import Modis, Conf
from MTGCNv3.args import args
from src.utils import EarlyStopping, accuracy

from pretrain.new_model import Prior
from pretrain.utils import pretrainConfig
    
    
def train_a_epoch(model, optimizer, data, train_mask, pseudo_mask, label):
    model.train()
    optimizer.zero_grad()
    logit, r = model(data.x, data.adj_t, drop_rate=args.pdr, backbone=args.backbone)
    if torch.sum(pseudo_mask) > 0:
        loss = F.cross_entropy(logit[train_mask], label[train_mask]) + args.plw*F.cross_entropy(logit[pseudo_mask], label[pseudo_mask])
    else:
        loss = F.cross_entropy(logit[train_mask], label[train_mask])
    loss.backward()
    print('the train loss is {}'.format(float(loss)), end=' ')
    optimizer.step()
    return logit
    
@torch.no_grad()
def test(model, data, train_mask, val_mask, test_mask, label):
    model.eval()
    logit, _ = model(data.x, data.adj_t, drop_rate=args.pdr, backbone=args.backbone)
    loss = F.cross_entropy(logit[val_mask], label[val_mask])
    print('the val loss is {}'.format(float(loss)))
    _, pred = logit.max(dim=1)
    train_correct = int(pred[train_mask].eq(data.y[train_mask]).sum().item())
    train_acc = train_correct / int(train_mask.sum())
    validate_correct = int(pred[val_mask].eq(data.y[val_mask]).sum().item())
    validate_acc = validate_correct / int(val_mask.sum())
    test_correct = int(pred[test_mask].eq(data.y[test_mask]).sum().item())
    test_acc = test_correct / int(test_mask.sum())
    return loss, train_acc, validate_acc, test_acc

    
def train_prior(data, label, train_mask, val_mask, test_mask, pseudo_mask, num_k, proir_layers=None):
    print(f'prior layer num:{proir_layers}')
    if proir_layers is None:
        proir_layers = args.proir_layers
    pretrain_cfg = pretrainConfig(args.backbone, data, args.p_hidden, proir_layers)
    model = Prior(pretrain_cfg).to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.plr, weight_decay=0.0005)
    epoch_num = args.num_epochs_max
    
    # mem = Modis(data.y, args.end_epoch_p-args.start_epoch_p, args.device)
    conf = Conf(data.y)
        
    ckpt_path = data_root+f'/prior_checkpoint/{args.dataset}_{args.backbone}_{args.train_spilt}.pt'
    early_stop = EarlyStopping(patience=100, path=ckpt_path)
    
    for epoch in range(500):
        logits = train_a_epoch(model, optimizer, data, train_mask, pseudo_mask, label)
        # if args.start_epoch_p <= epoch <= args.end_epoch_p:
        #     mem.update_memory(logits)
            
        val_loss, train_acc, val_acc, current_test_acc = test(model, data, train_mask, val_mask, test_mask, label)
        print('For the {} epoch, the train acc is {}, the val acc is {}, the test acc is {}.'.format(epoch, train_acc,
                                                                                                    val_acc,
                                                                                                    current_test_acc))
        early_stop(val_loss, val_acc, current_test_acc, model)
        if early_stop.early_stop:
            break
    print(f"early stop, final mtgcn test acc is {early_stop.best_test_acc}")
    
    model.load_state_dict(
        torch.load(ckpt_path, map_location=torch.device(args.device))
    )
    bst_logit, _ = model(data.x, data.adj_t, drop_rate=args.pdr, backbone=args.backbone)
    print(f'bst logit acc:{accuracy(bst_logit.max(-1)[1], data.y)}')
    easy_nodepool, n_train_mask = conf.get_easynode(bst_logit, num_k, train_mask, pseudo_mask)
    
    easy_idx = easy_nodepool['idx']
    
    torch.save(easy_nodepool, data_root+f'/prior_checkpoint/{args.dataset}_{args.backbone}_{args.train_spilt}_easynode.pt')
    print(f"easy node acc: {accuracy(easy_nodepool['pred'][easy_idx], data.y[easy_idx])}")
    
    print('Mission completes.')

    print('--------------------------------------------------------------------------')

    print('Dataset: {}.'.format(args.dataset))
    print('Backbone model: {}.'.format(args.backbone))
    del model
    del optimizer
    return early_stop.best_test_acc
    
if __name__ == '__main__':
    train_prior()