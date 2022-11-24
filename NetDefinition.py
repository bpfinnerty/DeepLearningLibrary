import projNet

class modelMLP():

    def __init__(self):
        
        self.inputLayer = projNet.NeuralNet(784,512)
        self.h1 = projNet.NeuralNet(512,256)
        self.h2 = projNet.NeuralNet(256,256)
        self.h3 = projNet.NeuralNet(256,128)
        self.outputLayer = projNet.NeuralNet(128,10)
        

    def forward(self, val):
        
        val = projNet.leakyRelu(self.inputLayer.ff(val))
        val = projNet.leakyRelu(self.h1.ff(val))
        val = projNet.leakyRelu(self.h2.ff(val))
        val = projNet.leakyRelu(self.h3.ff(val))
        val = projNet.softMax(self.outputLayer.ff(val))
        
        return val