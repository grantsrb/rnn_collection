import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from rnn.lstm import LSTM

"""
Combining the LSTM with concepts from Resisdual RNNs (https://arxiv.org/abs/1611.01457)
gives the following, where the only manipulation is adding old_h to the output h:

i = sigmoid(x.mm(W_x_i) + old_h.mm(W_h_i) + b_i)
f = sigmoid(x.mm(W_x_f) + old_h.mm(W_h_f) + b_f)
c = f*old_c + i*tanh(x.mm(W_x_c) + old_h.mm(W_h_c) + b_c)
o = sigmoid(x.mm(W_x_o) + old_h.mm(W_h_o) + b_o)
h = o*tanh(c) + old_h
"""

class ResLSTM(nn.Module):
    def __init__(self, x_size=256, state_size=256, layer_norm=False):
        super(ResLSTM, self).__init__()

        self.x_size = x_size
        self.state_size = state_size
        self.n_state_vars = 2
        self.lstm = LSTM(x_size, state_size, layer_norm=layer_norm)

    def forward(self, x, old_h, old_c):
        """
        x - New data coming into LSTM. FloatTensor Variable with shape (batch_size, state_size)
        old_h - short term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
        old_c - long term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)

        returns:
            h - new short term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
            c - new long term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
        """

        h,c = self.lstm(x, old_h, old_c)
        return h + old_h, c
