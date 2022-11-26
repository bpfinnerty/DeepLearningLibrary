import projNet

class modelMLP():

    def __init__(self):
        self.neuralNet = projNet.Net()
        self.neuralNet.addLayer(11,64)
        self.neuralNet.addLayer(64,32)
        self.neuralNet.addLayer(32,16)
        self.neuralNet.addLayer(16,16)
        self.neuralNet.addLayer(16,11)

    def forward(self, val):
        val = self.neuralNet.ff(val,0)

        val = self.neuralNet.Sigmoid(val)

        

        val = self.neuralNet.ff(val,1)

        val = self.neuralNet.Sigmoid(val)

        


        val = self.neuralNet.ff(val,2)

        val = self.neuralNet.Sigmoid(val)

        


        val = self.neuralNet.ff(val,3)

        val = self.neuralNet.Sigmoid(val)


        val = self.neuralNet.ff(val,4)

        val = self.neuralNet.softMax(val)

        
        return val
    
    def getNet(self):
        return self.neuralNet