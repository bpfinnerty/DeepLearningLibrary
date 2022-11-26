import projNet

class modelMLP():

    def __init__(self):
        self.neuralNet = projNet.Net()
        self.neuralNet.addLayer(2,2)
        self.neuralNet.addLayer(2,2)
        

    def forward(self, val):
        val = self.neuralNet.ff(val,0)
        print("Pre Activation 1")
        print(val)
        
        val = self.neuralNet.Sigmoid(val)
        print("Activation 1")
        print(val)
        
        val = self.neuralNet.ff(val,1)
        print("Pre Activation 2")
        print(val)
        val = self.neuralNet.Sigmoid(val)
        print("Activation 1")
        print(val)
        
        return val
    
    def getNet(self):
        return self.neuralNet