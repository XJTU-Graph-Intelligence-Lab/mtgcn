import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
# 数据集参数
# 'texas','wisconsin','cornell','washington'
# 'cora', 'citeseer', 'pubmed'
parser.add_argument('--dataset', type=str, default='texas')
parser.add_argument('--new_split', type=bool, default=True)
parser.add_argument('--train_spilt', type=float, default=5, help='训练集划分 [20, 0.48, 0.6]')
parser.add_argument('--split_id', type=int, default=1)

# 先验模型参数
parser.add_argument('-bb', '--backbone', type=str, default='GCN', help='The backbone model [GCN, GAT, APPNP, GCNII].')
parser.add_argument('--start_epoch_p', type=int, default=10, help='prior modis start epoch')
parser.add_argument('--end_epoch_p', type=int, default=50, help='prior modis start epoch')
parser.add_argument('--proir_layers', type=int, default=2, help='先验层数')
parser.add_argument('--pheads', type=int, default=2, help='prior GAT heads')
parser.add_argument('--pdr', type=float, default=0., help='prior dr')
parser.add_argument('--p_hidden', type=int, default=64, help='prior dr')
parser.add_argument('--alpha', type=float, default=0.1, help='The alpha value for APPNP (default: 0.1).')
parser.add_argument('--K', type=int, default=10, help='The K value for APPNP (default: 10).')

# 先验训练参数
parser.add_argument('--plr', type=float, default=0.01)
parser.add_argument('--plw', type=float, default=0.5, help='prior model pseudo label weight')

# 多轨道训练参数
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--tblr', type=float, default=0.005)
parser.add_argument('--tbwd', type=float, default=1e-4)
parser.add_argument('--tpwd', type=float, default=5e-5)
parser.add_argument('--lw', type=float, default=0.5, help='mtgnn model pseudo label weight')

# 多轨道模型参数
parser.add_argument('--ipt_conv', type=str, default='base')
parser.add_argument('--fai1', type=float, default=0.5, help='attn loss权重,半监督0.5')
parser.add_argument('--start_epoch', type=int, default=100, help='prior modis start epoch')
parser.add_argument('--end_epoch', type=int, default=300, help='prior modis start epoch')
parser.add_argument('--tau', type=float, default=1, help='温度系数')
parser.add_argument('--layer_num', type=int, default=16, help='轨道卷积层数')
parser.add_argument('--n_heads', type=int, default=2, help='sender attention')
parser.add_argument('--a', type=float, default=-0.5, help='track resudial param')
parser.add_argument('--dr', type=float, default=0.5, help='drop out rate')
parser.add_argument('--num_hidden', type=int, default=16)
parser.add_argument('--num_K', type=int, default=100)
parser.add_argument('--num_K_decay', type=int, default=30)

parser.add_argument('--num_epochs_patience', type=int, default=100)
parser.add_argument('--num_epochs_max', type=int, default=200)

parser.add_argument('--c_rounds', type=int, default=2)
parser.add_argument('--learning_rate_decay_patience', type=int, default=100)
parser.add_argument('--learning_rate_decay_factor', type=float, default=0.8)
args = parser.parse_args()