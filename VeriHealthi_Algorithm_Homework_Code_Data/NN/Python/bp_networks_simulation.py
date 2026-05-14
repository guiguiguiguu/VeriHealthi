#!/usr/bin/python
#
# Copyright (c) 2026, VeriSilicon Holdings Co., Ltd. All rights reserved
#
# 1. Redistributing the source code of this software is only allowed after
# receiving explicit, written permission from VeriSilicon. The copyright notice,
# this list of conditions and the following disclaimer must be retained in all
# source code distributions.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.

# simple BP nerual networks for classification

from turtle import forward
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cmath import pi
import pandas as pd

torch.set_printoptions(precision=10)

def create_dataset(N=100, K=2):
    N = 100
    K = 2
    x = torch.zeros((N*K, 2), dtype=torch.float)
    y = torch.zeros(N*K, dtype=torch.long)
    for i in range(K):
        ix = range(i*N, (i+1)*N)
        d = np.array([i, 0.75*(i % 2)-0.25])
        r = 1 + np.random.rand(N)*0.5
        t = np.linspace(i*pi+0.5, i*pi+0.5+pi, N) + np.random.rand(N)*0.2
        x[ix] = torch.tensor(
            d + np.c_[r*np.cos(t), r*np.sin(t)], dtype=torch.float)
        y[ix] = torch.tensor(i, dtype=torch.long)
    return x, y


class BPNetwork(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(BPNetwork, self).__init__()
        self.input_layer = nn.Linear(n_input, n_hidden)
        self.hidden_layer = nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.input_layer(input)
        out = F.relu(out)
        out = self.hidden_layer(out)
        return out


if __name__ == "__main__":
    x, y = create_dataset(100, 2)
    print(x, y)
    pd.DataFrame(x.numpy()).to_csv('feature.csv', header=None, index=None)
    pd.DataFrame(y.numpy()).to_csv('label.csv', header=None, index=None)
    bp_network = BPNetwork(2, 10, 2)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(bp_network.parameters(), lr=1e-1, momentum=0.9)

    for t in range(30000):
        y_pred = bp_network.forward(x)

        loss = loss_func(y_pred, y)
        # print(t, loss.item())
        if t % 2000 == 1999:
            print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Model structure: {bp_network}\n\n")
    # file = open('bp_networks.txt','a')
    for name, param in bp_network.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param} \n")
        # file.write(f"Layer: {name} | Size: {param.size()} | Values : {param} \n")
    # file.close()

    max_x = torch.max(x[:, 0])
    min_x = torch.min(x[:, 0])
    max_y = torch.max(x[:, 1])
    min_y = torch.min(x[:, 1])

    xd = torch.linspace(min_x, max_x, 101)
    yd = torch.linspace(min_y, max_y, 101)

    xt = [(i, j) for i in xd for j in yd]
    xt = torch.tensor(xt)
    y_pred = bp_network.forward(xt)
    p = F.softmax(y_pred, dim=1)
    yt = torch.argmax(p, dim=1)

    plt.figure(1)
    for i in range(len(x)):
        if y[i] == 0:
            c_color = 'ro'
        else:
            c_color = 'bo'
        plt.plot(x[i, 0], x[i, 1], c_color)
    plt.grid(True)

    plt.figure(2)
    for i in range(len(xt)):
        if yt[i] == 0:
            c_color = 'y.'
        else:
            c_color = 'm.'
        plt.plot(xt[i, 0], xt[i, 1], c_color)
    for i in range(len(x)):
        if y[i] == 0:
            c_color = 'ro'
        else:
            c_color = 'bo'
        plt.plot(x[i, 0], x[i, 1], c_color)
    plt.grid(True)
    plt.show()
