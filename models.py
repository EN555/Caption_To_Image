import torch.nn as nn
import torch.nn.functional as F


class encodeImage(nn.Module):
    def __init__(self, embedded_size):
        conv1 = F.conv2d()
        max_pool = F.max_pool2d()
        self.dropout = F.dropout()




class decodeImage(nn.Module):
    pass