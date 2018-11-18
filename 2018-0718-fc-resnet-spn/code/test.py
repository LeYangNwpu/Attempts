'''
Ask for help:
    there are two problems when I run the demo pytorch_spn
    https://github.com/Liusifei/pytorch_spn
    the problems are illustrated as following.
My environment:
    python = 2.7
    pytorch = 0.2.0
    I can successfully install this project.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from pytorch_spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind

class Refine(nn.Module):

    def __init__(self):
        super(Refine, self).__init__()
        self.Propagator = GateRecurrent2dnoind(True, True)

    def forward(self, X):

        G1 = Variable(torch.randn(1, 3, 10, 10))
        G2 = Variable(torch.randn(1, 3, 10, 10))
        G3 = Variable(torch.randn(1, 3, 10, 10))

        X = X.cuda()
        G1 = G1.cuda()
        G2 = G2.cuda()
        G3 = G3.cuda()

        sum_abs = G1.abs() + G2.abs() + G3.abs()
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        G1_norm = torch.div(G1, sum_abs)
        G2_norm = torch.div(G2, sum_abs)
        G3_norm = torch.div(G3, sum_abs)

        G1 = torch.add(-mask_need_norm, 1) * G1 + mask_need_norm * G1_norm
        G2 = torch.add(-mask_need_norm, 1) * G2 + mask_need_norm * G2_norm
        G3 = torch.add(-mask_need_norm, 1) * G3 + mask_need_norm * G3_norm

        output = self.Propagator.forward(X, G1, G2, G3)

        return output

X = Variable(torch.randn(1,3,10,10))
X = X.cuda()

model = Refine()
# optimizer = optim.SGD(model.parameters())
'''
Question 1:
    model.parameters() is an empty list.
    Does it mean that no parameter to optimize?
'''
print('  + Number of params: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])))
output = model(X)

target = Variable(torch.randn(1,3,10,10))
target = target.cuda()

'''
Question2:
    loss.backward fails  
    error information 'there are no graph nodes that require computing gradients'
'''
mseLoss = torch.nn.MSELoss()
loss = mseLoss(output, target)
# optimizer.zero_grad()
loss.backward()
# optimizer.step()
