import projNet

class modelMLP():

    def __init__(self):
        self.neuralNet = projNet.Net()
        self.neuralNet.addLayer(784,16)
        self.neuralNet.addLayer(16,16)
        self.neuralNet.addLayer(16,16)
        self.neuralNet.addLayer(16,16)
        self.neuralNet.addLayer(16,10)
        

    def forward(self, val):
        print("First layer")
        #self.neuralNet.printDim(0)
        val = self.neuralNet.ff(val,0)
        print("Time for Relu")
        val = self.neuralNet.leakyRelu(val)
        
        print("Second Layer")
        #self.neuralNet.printDim(1)
        val = self.neuralNet.ff(val,1)
        val = self.neuralNet.leakyRelu(val)
        
        print("Third Layer")
        #self.neuralNet.printDim(2)
        val = self.neuralNet.ff(val,2)
        val = self.neuralNet.leakyRelu(val)
        
        print("Fourth layer")
        #self.neuralNet.printDim(3)
        val = self.neuralNet.ff(val,3)
        val = self.neuralNet.leakyRelu(val)
        
        print("fifth layer")
        #self.neuralNet.printDim(4)
        val = self.neuralNet.ff(val,4)
        print("Time for Soft Max")
        val = self.neuralNet.softMax(val)
        
        return val
    
    def getNet(self):
        return self.neuralNet