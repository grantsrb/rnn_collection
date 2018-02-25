import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from rnn import LSTM, NestedLSTM, DoubleGRU, ResLSTM, ResNestedLSTM, GRU
import sys
import matplotlib.pyplot as plt

"""
This script is used to test the various RNNs on a simple binary number prediction task.
The specified RNN type is used to predict a string of random binary digits. The RNNs
should be able to easily overfit to the random string.
"""

batch_size = 1
state_size = 30
lr = 0.01
x_size = 1
seq_len = 200
n_epochs = 1000
n_trials = 1 # Number of trials per rnn type
layer_norm=False
rnn_types = ["ResLSTM", "NestedLSTM", "LSTM", "GRU"]
rnn_loss_logs = dict()

binary = np.random.randint(0,x_size+1,size=(1,seq_len))
X = np.repeat(binary[:,:-1], batch_size, 0)
Y = np.repeat(binary[:,1:], batch_size, 0)
binary = binary.squeeze()

X = Variable(torch.FloatTensor(X))
Y = Variable(torch.LongTensor(Y))

def pred_fxn(rnn, state_vars, classifier, x):
    state_vars = rnn(x.unsqueeze(1), *state_vars)
    prediction = state_vars[0].mm(classifier)
    return prediction, state_vars

def update(classifier, lr):
    classifier.data = classifier.data - lr*classifier.grad.data
    return Variable(classifier.data, requires_grad=True)

means = torch.zeros(state_size, 2)
rand_vector = torch.normal(means, std=1/float(np.sqrt(state_size)))
lossfxn = nn.CrossEntropyLoss()
loss = 0
figcount = 1
for rnn_type in rnn_types:
    print("RNN TYPE:", rnn_type)
    loss_logs = []
    for trial in range(n_trials):
        loss_log = []
    
        if 'ResNestedLSTM' == rnn_type: rnn = ResNestedLSTM(x_size, state_size, layer_norm=layer_norm)
        elif 'ResLSTM' == rnn_type: rnn = ResLSTM(x_size, state_size, layer_norm=layer_norm)
        elif 'ResRNN' == rnn_type: rnn = ResRNN(x_size, state_size, layer_norm=layer_norm)
        elif 'NestedLSTM' == rnn_type: rnn = NestedLSTM(x_size, state_size, layer_norm=layer_norm)
        elif 'LSTM' == rnn_type: rnn = LSTM(x_size, state_size, layer_norm=layer_norm)
        elif 'DoubleGRU' == rnn_type: rnn = DoubleGRU(x_size, state_size, layer_norm=layer_norm)
        elif 'GRU' == rnn_type: rnn = GRU(x_size, state_size, layer_norm=layer_norm)
        elif "ResGRU" == rnn_type: rnn = ResGRU(x_size, state_size, layer_norm=layer_norm)
        adam = optim.SGD(rnn.parameters(), lr=lr)
        adam.zero_grad()
        classifier = Variable(rand_vector.clone(), requires_grad=True)
    
        for i in range(n_epochs):
            X, Y = Variable(X.data), Variable(Y.data)
            state_vars = [Variable(torch.zeros(batch_size,state_size)) for i in range(rnn.n_state_vars)]
            for j in range(X.shape[1]):
                x,y = X[:,j], Y[:,j]
                prediction, state_vars = pred_fxn(rnn, state_vars, classifier, x)
                loss += lossfxn(prediction, y)
        
            loss.backward()
            adam.step()
            adam.zero_grad()
            classifier = update(classifier, lr)
            print("Epoch:", i, "–– LOSS:", loss.data[0])
            loss_log.append(loss.data[0])
            loss = 0
        loss_logs.append(loss_log)
    
    loss_logs = np.asarray(loss_logs, dtype=np.float32)
    loss_log = np.mean(loss_logs, axis=0)
    plt.figure(figcount)
    figcount+=1
    plt.plot(np.arange(n_epochs), loss_log)
    plt.title(rnn_type+" Loss Chart")
    plt.xlabel("Epoch")
    plt.ylabel("Prediction Cross Entropy")
    plt.legend((rnn_type,),loc='upper right')
    plt.savefig(rnn_type+"_fig.png")
    rnn_loss_logs[rnn_type] = loss_log
    
    X, Y = Variable(X.data), Variable(Y.data)
    state_vars = [Variable(torch.zeros(batch_size,state_size)) for i in range(rnn.n_state_vars)]
    pred_seq = np.zeros_like(binary[1:])
    for i in range(X.shape[1]):
        x,y = X[:,i], Y[:,i]
        prediction, state_vars = pred_fxn(rnn, state_vars, classifier, x)
        _, p = torch.max(prediction,1)
        pred_seq[i] = p.data.squeeze()[0]
    print("Acc:", np.mean(np.equal(pred_seq, binary[1:])))

plt.figure(figcount)
figcount+=1
for rtype in rnn_types:
    plt.plot(rnn_loss_logs[rtype], label=rtype)
plt.title("Loss Comparison Chart")
plt.xlabel("Epoch")
plt.ylabel("Prediction Cross Entropy")
plt.legend(loc='lower left')
plt.savefig("full_comparison_fig.png")

plt.figure(figcount)
figcount+=1
for rtype in rnn_types:
    plt.plot(rnn_loss_logs[rtype], label=rtype)
plt.title("Loss Comparison Chart")
plt.xlabel("Epoch")
plt.ylabel("Prediction Cross Entropy")
plt.ylim(0,250)
plt.legend(loc='lower left')
plt.savefig("comparison_fig.png")
