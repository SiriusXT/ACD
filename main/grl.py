import math
from re import X
from statistics import mode
import torch.nn as nn
import torch.nn.functional as F
import torch


class DLM(nn.Module):
    def __init__(self):
        super(DLM, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1) * 1, requires_grad=True)
        # self.ins_T = nn.Parameter(torch.ones(1) * 1, requires_grad=True)
        self.mlp1 = nn.Linear(10 , 1 )
        # nn.init.constant_(self.mlp1.weight, 0)
        # nn.init.constant_(self.mlp1.bias, 0)
        # self.mlp2 = nn.Linear(10, 1)
        self.grl = GradientReversal()
        self.reset_parameters()


    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # nn.init.constant_(self.mlp1.weight, 0)
        # nn.init.constant_(self.mlp1.bias, -10)
        self.lr=nn.LeakyReLU(0.1)

    def forward(self, input1, input2, lambda_=1):
        # return self.grl(self.alpha, lambda_)
        input = self.mlp1(torch.cat([input1, input2], -1))
        input = torch.relu(input)
        return self.grl(input, lambda_)


from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()
        # self.lambda_ = lambda_

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)

