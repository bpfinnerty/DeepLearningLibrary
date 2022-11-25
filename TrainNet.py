import pandas as pd
import projNet
import numpy as np
from NetDefinition import *


def oneHotEncode(vector,numClasses):
    
    zeroed = np.zeros((vector.size, numClasses))
    encoded = zeroed[np.arange(vector.size),vector] =1
    return encoded

def main(learningRate, numThreads, epoch, batch_size, trainDataPath,testDataPath, model):
    
    train_df = pd.read_csv(trainDataPath,header=None,skiprows=1)
    print("opened data")
    
    model.neuralNet.setLearningRate(learningRate)
    projNet.setThreads(numThreads)
    
    model.getNet().printDim(0)
    model.getNet().printDim(1)
    model.getNet().printDim(2)
    model.getNet().printDim(3)
    model.getNet().printDim(4)
    print("Set initial values")
    
    for e in range(0,epoch):
        print("epoch: " + str(e))
        
        shuffled = train_df.sample(frac=.1)
        truth = shuffled.iloc[:,0].to_numpy()
        train = shuffled.iloc[:,1:].to_numpy()
        encoded_truth = oneHotEncode(truth,10)
        
        
        max_examples = train.size
        index = 0
        avgLoss = 0
        print("ready for loop")
        while index < max_examples:
            print(index)

            model.neuralNet.zeroGrad()
            print("After grad")
            counter = 0
            while counter < batch_size:
                if counter >= max_examples:
                    break
                print(counter)
                predictions = model.forward(train[index])
                loss = model.neuralNet.crossEntropy(predictions, encoded_truth)
                avgLoss += loss
                model.neuralNet.backwardStep(predictions,encoded_truth)
                index+=1
                counter+=1
            print("finished batch")
            model.neuralNet.updateWeights()
        print("Avg Loss: " + str(avgLoss/max_examples) + "for epoch: " + str(e))    
        
        
        
    
    

if __name__ == "__main__":
    
    learningRate = 0.01
    numThreads = 1
    epoch = 30
    batch = 16
    
    trainingDataPath = "mnist_train.csv"
    testingDataPath = "mnist_test.csv"
    
    model = modelMLP()
    
    main(learningRate, numThreads, epoch, batch, trainingDataPath, testingDataPath, model)
    