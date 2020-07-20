import torch
import torch.nn as nn

class Resize(nn.Module):
    def __init__(self, size):
        super(Resize, self).__init__()
        self.size = size
    
    def forward(self, tensor):
        return tensor.view(self.size)
    

class Flatten(nn.Module):
    def forward(self, tensor):
        shape = tensor.shape[-3:]
        return tensor.view((-1, shape[0]*shape[1]*shape[2]))
    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
