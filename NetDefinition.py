import projNet

class modelMLP():

    def __init__(self):
        self.neuralNet = projNet.Net()
        self.neuralNet.addLayer(784,128)
        self.neuralNet.addLayer(128,64)
        self.neuralNet.addLayer(64,32)
        self.neuralNet.addLayer(32,16)
        self.neuralNet.addLayer(16,10)
        

    def forward(self, val):
        print("First layer")
        val = self.neuralNet.getLayer(0).ff(val)
        val = self.neuralNet.leakyRelu(val)
        
        print("Second Layer")
        val = self.neuralNet.getLayer(1).ff(val)
        val = self.neuralNet.leakyRelu(val)
        
        print("Third Layer")
        val = self.neuralNet.getLayer(2).ff(val)
        val = self.neuralNet.leakyRelu(val)
        
        print("Fourth layer")
        val = self.neuralNet.getLayer(3).ff(val)
        val = self.neuralNet.leakyRelu(val)
        
        print("fifth layer")
        val = self.neuralNet.getLayer(4).ff(val)
        val = self.neuralNet.softMax(val)
        
        return val