import projNet

class modelMLP():

    def __init__(self):
        self.neuralNet = projNet.Net()
        self.neuralNet.addLayer(2,2)
        self.neuralNet.addLayer(2,3)
        

    def forward(self, val):
        val = self.neuralNet.ff(val,0)
        val = self.neuralNet.relu(val)
        
        val = self.neuralNet.ff(val,1)
        print("PreSoftMax")
        print(val)
        val = self.neuralNet.softMax(val)
        
        return val
    
    def getNet(self):
        return self.neuralNet