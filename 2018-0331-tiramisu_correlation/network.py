import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim

import utils as utils


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=48, kernel_size=3, padding=1, stride=1, bias=True)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm2d(num_features=48)
        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pad(x)
        out = self.norm(self.relu(self.conv(x)))
        out = self.pool(out)
        return out

model = Network()
model = model.cuda()
x = torch.randn(2, 3, 127, 127)
x = Variable(x.cuda())
out = model(x)
print(out.size())


