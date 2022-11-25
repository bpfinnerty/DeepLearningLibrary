import pandas as pd
import projNet
import numpy
from NetDefinition import *


def oneHotEncode(vector,numClasses):
    
    zeroed = np.zeros((vector.size, numClasses))
    encoded = zeroed[np.arange(vector.size),vector] =1
    return encoded

def main(learningRate, numThreads, epoch, batch_size, trainDataPath,testDataPath, model):
    train_df = pd.read_csv(trainDataPath,header=None,skiprows=1)
    projNet.setLearningRate(learningRate)
    projNet.setThreads(numThreads)
    
    
    for e in range(0,epoch):
        shuffled = train_df.sample(frac=.1)
        truth = shuffled.iloc[:,0].to_numpy()
        train = shuffled.iloc[:,1:].to_numpy()
        encoded_truth = oneHotEncode(truth)
        max_examples = train.truth
        index = 0
        avgLoss = 0
        while index < max_examples:
            
            projNet.zeroGrad()
            counter = 0
            while counter < batch_size:
                if counter >= max_examples:
                    break
                
                predictions = model.forward(train[index])
                loss = projNet.crossEntropy(predictions, encoded_truth[index])
                avgLoss += loss
                projNet.backwardStep(predictions,encoded_truth[index])
            print("finished batch")
            projNet.updateWeights()
        print("Avg Loss: " + str(avgLoss/max_examples) + "for epoch: " + str(e))    
        
        
        
    
    

if __name__ == "__main__":
    
    learningRate = 0.01
    numThreads = 8
    epoch = 30
    batch = 16
    
    trainingDataPath = "mnist_train.csv"
    testingDataPath = "mnist_test.csv"
    
    model = modelMLP()
    
    main(learningRate, numThreads, epoch, batch, trainingDataPath, testingDataPath, model)
    