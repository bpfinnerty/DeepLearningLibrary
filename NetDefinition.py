import projNet

class modelMLP():

    def __init__(self):
        self.neuralNet = projNet.Net()
        self.neuralNet.addLayer(784,16)
        self.neuralNet.addLayer(16,16)
        self.neuralNet.addLayer(16,10)
        

    def forward(self, val):
        #print("First layer")
        #self.neuralNet.printDim(0)
        #print(val)
        val = self.neuralNet.ff(val,0)
        #print(val)
        #print("Time for sigmoid")
        val = self.neuralNet.Sigmoid(val)
        #print(val)
        
        #print("Second Layer")
        #self.neuralNet.printDim(1)
        val = self.neuralNet.ff(val,1)
        #print(val)
        val = self.neuralNet.Sigmoid(val)
        #print(val)
        
        #print("Third Layer")
        #self.neuralNet.printDim(2)
        #val = self.neuralNet.ff(val,2)
        #print(val)
        #val = self.neuralNet.Sigmoid(val)
        #print(val)
        
        #print("Fourth layer")
        #self.neuralNet.printDim(3)
        #val = self.neuralNet.ff(val,3)
        #print(val)
        #val = self.neuralNet.Sigmoid(val)
        #print(val)
        
        #print("fifth layer")
        #self.neuralNet.printDim(4)
        val = self.neuralNet.ff(val,2)
        #print("Time for Soft Max")
        #print(val)
        val = self.neuralNet.softMax(val)
        #print(val)
        
        return val
    
    def getNet(self):
        return self.neuralNet