import numpy as np
from scipy import sparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from collections import Counter
from torch_geometric.datasets import Planetoid, WebKB, Coauthor, Amazon
import torch_geometric.transforms as T
from glob import glob
import operator
import os

path_list = os.path.abspath(__file__).split('/')[:-2]
symbol = "/"
root_path = symbol.join(path_list) + '/data'

def load_data(data_name, new_split=False, train_spilt=0.6, device='cuda:0'):
    #
    transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    if data_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root_path, name=data_name, transform=transform)
    elif data_name in  ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root_path, name=data_name, transform=transform)
    elif data_name in ['cs', 'physics']:
        dataset = Coauthor(root_path, name=data_name, transform=transform)
    elif data_name in ['computers', 'photo']:
        dataset = Amazon(root_path, name=data_name, transform=transform)
    data = dataset[0]
    if new_split:
        new_trn, new_val, new_tst = [], [], []
        split_paths = glob(root_path + f'/new_splits/{data_name}_split_{train_spilt}_*.npz')
        for path in split_paths:
            split = np.load(path)
            new_trn.append(split['train_mask'])
            new_val.append(split['val_mask'])
            new_tst.append(split['test_mask'])
        data.train_mask = torch.from_numpy(np.array(new_trn).T)
        data.val_mask = torch.from_numpy(np.array(new_val).T)
        data.test_mask = torch.from_numpy(np.array(new_tst).T)
    if data_name in ['cs', 'physics', 'computers', 'photo']:
        if os.path.exists(root_path + f'/new_splits/{data_name}_split_20.npz'):
            split = np.load(root_path + f'/new_splits/{data_name}_split_20.npz')
            data.train_mask = torch.from_numpy(split['train_mask'])
            data.val_mask = torch.from_numpy(split['val_mask'])
            data.test_mask = torch.from_numpy(split['test_mask'])
        else:
            data.train_mask, data.val_mask, data.test_mask = get_node_spilt(data_name, data.x.shape[0], data.y)
    return data.to(device)


def change_spilt(data_name, node_num, train_p, val_p, i):
    train_mask = np.zeros(node_num, dtype=bool)
    val_mask = np.zeros(node_num, dtype=bool)
    test_mask = np.zeros(node_num, dtype=bool)

    train_node_num = int(node_num * train_p)
    val_num = int(node_num * val_p)

    sample_index = np.zeros(node_num)
    sample_index[:train_node_num] = 1  # 训练集
    sample_index[train_node_num:train_node_num + val_num] = -1  # 验证集
    np.random.shuffle(sample_index)

    train_mask[np.where(sample_index == 1)[0]] = True
    val_mask[np.where(sample_index == -1)[0]] = True
    test_mask[np.where(sample_index == 0)[0]] = True
    np.savez(root_path + f'/new_splits/{data_name}_split_{train_p}_{i}.npz', train_mask=train_mask, val_mask=val_mask,
             test_mask=test_mask)



def get_node_spilt(data_name, node_num, label):
    """
    构造半监督节点分类数据集
    :param node_num:
    :param label:
    :return:
    """
    train_mask = np.zeros(node_num, dtype=bool)
    val_mask = np.zeros(node_num, dtype=bool)
    test_mask = np.zeros(node_num, dtype=bool)

    label = label.cpu().numpy()
    num_class = max(label)+1
    for c in range(num_class):
        c_idx = np.where(label == c)[0]
        n_per_c = min(c_idx.shape[0], 5)
        choice_idx = np.random.choice(c_idx, n_per_c, replace=False)
        train_mask[choice_idx] = True

    no_train_idx = np.where(~train_mask)[0]
    # no_train_idx = np.random.choice(no_train_idx, 1500, replace=False)
    test_mask[no_train_idx[:100]] = True
    val_mask[no_train_idx[100:]] = True
    
    np.savez(root_path + f'/new_splits/{data_name}_split_5_0.npz', train_mask=train_mask, val_mask=val_mask,
             test_mask=test_mask)
    return torch.from_numpy(train_mask), torch.from_numpy(val_mask), torch.from_numpy(test_mask)


def normalize(mx):
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()

    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(pred, labels):
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_lb(labels, train_mask, priori):
    lb = F.one_hot(labels).type(torch.float32)
    lb[~train_mask] = priori[~train_mask]
    return lb.detach()


def update_pseudo(data, output, t):
    track_m, lp, linear_h = output
    track_m = F.softmax(track_m, 1)
    lp = F.softmax(lp, 1)
    linear_h = F.softmax(linear_h, 1)

    # vote
    m_conf, m_pred = torch.max(track_m, 1)
    lp_conf, lp_pred = torch.max(lp, 1)
    linear_conf, linear_pred = torch.max(linear_h, 1)

    pred = torch.stack([m_pred, lp_pred, linear_pred]).detach().cpu().numpy()
    max_conf, max_idx = torch.max(torch.stack([m_conf, lp_conf, linear_conf]), 1)

    node_num = pred.shape[0]
    send_num, true_num = 0, 0
    for i in range(node_num):
        vote_dict = dict(sorted(Counter(pred[:, i]).items(), key=operator.itemgetter(1)))
        if vote_dict[list(vote_dict.keys())[-1]] >= 2:
            data.pseudo_mask[i] = True
            data.pseudo_label[i] = int(list(vote_dict.keys())[-1])
            send_num += 1
        elif max_conf[i] > t:
            data.pseudo_mask[i] = True
            data.pseudo_label[i] = max_idx[i]
            send_num += 1

        if data.pseudo_label[i] == data.labels[i]:
            true_num += 1
    print(f'send num: {send_num}, send_acc: {true_num/send_num}')

if __name__ == '__main__':
    path = os.path.abspath(__file__)
    name_list = ['cora', 'citeseer', 'pubmed', 'cornell', 'texas', 'wisconsin']
    for data_name in tqdm(name_list):
        data = load_data(data_name)
        for i in range(10):
            change_spilt(data_name, data.x.shape[0], 0.48, 0.32, i)