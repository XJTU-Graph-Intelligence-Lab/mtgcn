import math

from torch import Tensor
from args import args

def glorot(tensor: Tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor: Tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor: Tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def rand_zero_to_ones(tensor: Tensor):
    if tensor is not None:
        tensor.data.uniform_(0, 1)


def average_agg(tensor: Tensor):
    result = 0
    batch_number = tensor.size(0)
    for i in range(batch_number):
        result += tensor[i]
    result /= batch_number
    return result


class pretrainConfig:
    def __init__(self, backbone, data, hidden, proir_layers):
        self.w = args.alpha
        self.K =  int(args.K)
        self.pheads = args.pheads
        self.pdr = args.pdr
        self.num_layers = proir_layers
        self.is_bns = False
        self.backbone = backbone
        self.in_channels, self.out_channels = self.get_dim(data)
        self.hidden_channels = hidden
    
    @staticmethod
    def get_dim(data):
        return data.x.shape[1], data.y.max().item() + 1        