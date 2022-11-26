import torch.nn as nn
import torch
import torch.nn.functional as f

class MLP(nn.Module):


    def __init__(self):
        super(MLP, self).__init__()
        self.inputLayer = nn.Linear(784,128)
        # self.h1 = nn.Linear(256,128)
        # self.h2 = nn.Linear(128,64)
        self.h3 = nn.Linear(128,64)
        self.outputLayer = nn.Linear(64,10)

    def forward(self, val):
        # implement your code here:
        # first layer of convolution with a max pooling
        val = self.inputLayer(val)
        val = torch.sigmoid(val)

        # second layer of convolution with a max pooling
        # val = self.h1(val)
        # val = torch.sigmoid(val)

        # third layer of convolution with a max pooling
        # val = self.h2(val)
        # val = torch.sigmoid(val)
        val = self.h3(val)
        val = torch.sigmoid(val)
        val = self.outputLayer(val)
        val = f.softmax(val,dim=1)
        return val