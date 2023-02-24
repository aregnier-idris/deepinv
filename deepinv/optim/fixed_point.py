import torch
import torch.nn as nn
from deepinv.optim.utils import check_conv

class FixedPoint(nn.Module):
    '''
    '''
    def __init__(self, iterator, max_iter=50, early_stop=True, crit_conv=None, verbose=False) :
        super().__init__()
        self.iterator = iterator 
        self.max_iter = max_iter
        self.crit_conv = crit_conv
        self.verbose = verbose
        self.early_stop = early_stop

    def forward(self, init, *args):
        x = init
        for it in range(self.max_iter):
            x_prev = x
            x = self.iterator(x, it, *args)
            if self.early_stop and check_conv(x_prev, x, it, self.crit_conv, self.verbose) :
                break
        return x


