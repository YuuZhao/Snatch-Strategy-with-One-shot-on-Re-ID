from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init


class APMLoss(nn.Module):
    def __init__(self):
        super(APMLoss, self).__init__()
        self.loss_g = nn.CrossEntropyLoss()
        self.loss_l1 = nn.CrossEntropyLoss()
        self.loss_l2 = nn.CrossEntropyLoss()
        self.loss_l3 = nn.CrossEntropyLoss()
        self.loss_l4 = nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss_g = self.loss_g(input[0], target)
        loss_l1 = self.loss_l1(input[1], target)
        loss_l2 = self.loss_l2(input[2], target)
        loss_l3 = self.loss_l3(input[3], target)
        loss_l4 = self.loss_l4(input[4], target)
        return loss_g+loss_l1+loss_l2+loss_l3+loss_l4
