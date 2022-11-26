import torch.nn as nn
import torch
import torch.nn.functional as f

class MLP(nn.Module):


    def __init__(self):
        super(MLP, self).__init__()
        self.inputLayer = nn.linear(784,256)
        self.h1 = nn.linear(256,128)
        self.h2 = nn.linear(126,64)
        self.h3 = nn.linear(64,32)
        self.outputLayer = nn.linear(32,10)

    def forward(self, val):
        # implement your code here:
        # first layer of convolution with a max pooling
        val = f.sigmoid(self.inputLayer(val))

        # second layer of convolution with a max pooling
        val = f.sigmoid(self.h1(val))

        # third layer of convolution with a max pooling
        val = f.sigmoid(self.h2(val))
        val = f.sigmoid(self.h3(val))
        val = f.softmax(self.outputLayer(val))
        return val