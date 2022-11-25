import projNet

class modelMLP():

    def __init__(self):
        
        self.inputLayer = projNet.NeuralNet(784,512)
        self.h1 = projNet.NeuralNet(512,256)
        self.h2 = projNet.NeuralNet(256,256)
        self.h3 = projNet.NeuralNet(256,128)
        self.outputLayer = projNet.NeuralNet(128,10)
        

    def forward(self, val):
        print("First layer")
        val = projNet.leakyRelu(self.inputLayer.ff(val))
        
        print("Second Layer")
        val = projNet.leakyRelu(self.h1.ff(val))
        
        print("Third Layer")
        val = projNet.leakyRelu(self.h2.ff(val))
        
        print("Fourth layer")
        val = projNet.leakyRelu(self.h3.ff(val))
        
        print("fifth layer")
        val = projNet.softMax(self.outputLayer.ff(val))
        
        return val