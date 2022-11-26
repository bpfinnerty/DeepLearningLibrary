import projNet

class modelMLP():

    def __init__(self):
        self.neuralNet = projNet.Net()
        self.neuralNet.addLayer(784,256)
        self.neuralNet.addLayer(256,128)
        self.neuralNet.addLayer(128,64)
        self.neuralNet.addLayer(64,32)
        self.neuralNet.addLayer(32,10)

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