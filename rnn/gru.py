import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np

"""
GRU units follow the formulae:

z = sigmoid(W_x_z.mm(x) + W_h_z.mm(old_h) + b_z)
r = sigmoid(W_x_r.mm(x) + W_h_r.mm(old_h) + b_r
h = z*old_h + (1-z)*tanh(W_x_h.mm(x) + W_h_h.mm(r*old_h) + b_h)

Where x is the new, incoming data and old_h is the h at the previous time step.
Each of the W_x_ and W_h_ terms are weight matrices and the b_ terms are biases.
In this implementation, all of the W_x_ terms are combined into a single variable. Same
with the W_h_ and b_ terms.
"""

class GRU(nn.Module):
    def __init__(self, x_size=256, state_size=256, layer_norm=False):
        super(GRU, self).__init__()

        self.x_size = x_size
        self.state_size = state_size
        self.n_state_vars = 1

        # Internal GRU Entry Parameters
        means = torch.zeros(3, x_size, state_size)
        self.W_x1 = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(state_size))), requires_grad=True)

        # Internal GRU State Parameters
        means = torch.zeros(3, state_size, state_size)
        self.W_h = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(state_size))), requires_grad=True)

        # Internal GRU Bias Parameters
        self.b = nn.Parameter(torch.zeros(3,1,state_size), requires_grad=True)

        # Non Linear Activation Functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, old_h):
        """
        old_h - running state of GRU. FloatTensor Variable with shape (batch_size, state_size)
        x - New data coming into GRU. FloatTensor Variable with shape (batch_size, state_size)
        """

        z = self.sigmoid(x.mm(self.W_x1[0]) + old_h.mm(self.W_h[0]) + self.b[0])
        r = self.sigmoid(x.mm(self.W_x1[1]) + old_h.mm(self.W_h[1]) + self.b[1])
        h = z*old_h + (1-z)*self.tanh(x.mm(self.W_x1[2]) + (r*old_h).mm(self.W_h[2]) + self.b[2])
        return (h,)

    def check_grads(self):
        for p in list(self.parameters()):
            if torch.sum(p.grad.data != p.grad.data):
                print("NaN in grads")
