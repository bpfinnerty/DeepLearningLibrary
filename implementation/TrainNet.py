import pandas as pd
import projNet
import numpy as np
from NetDefinition import *


def oneHotEncode(vector,numClasses):
    #print("One Hot Encode\n")
    encoded_array = np.zeros((vector.size, numClasses), dtype=int)
    encoded_array[np.arange(vector.size),vector] = 1 
    return encoded_array

def testAccuracy(model, test_df):
    # print("In Test Accuracy\n")
    
    
    # print("convert to numpy")
    truth_subset = test_df.iloc[:,0].to_numpy()
    # print(truth_subset)
    test_subset = test_df.iloc[:,1:].to_numpy()/255
    print(test_subset.shape)
    
    # print("Get shape")
    max_examples = test_subset.shape[0]
    # print(max_examples)
    
    index = 0
    totalCorrect = 0
    # print("ready for loop")
    
    while index < max_examples:
        predictions = model.forward(test_subset[index])
        # print("Predicted Value: " +  str(np.argmax(predictions)))
        # print("Actual: " + str(truth_subset[index]))
        if(np.argmax(predictions) == truth_subset[index]):
            totalCorrect+=1
        index+=1
        
    print("Test accuracy: " + str(totalCorrect/max_examples))
        


def main(learningRate, numThreads, epoch, batch_size, trainDataPath,testDataPath, model):
    
    train_df = pd.read_csv(trainDataPath,header=None,skiprows=1)
    t_df = pd.read_csv(testDataPath,header=None,skiprows=1)
    # print(df)
   
    # print(test_df.shape)
    #print("opened data")
    
    model.neuralNet.setLearningRate(learningRate)
    model.neuralNet.setThreadNum(numThreads)

    #print("Set initial values")
    
    for e in range(0,epoch):
        print("epoch: " + str(e))
        #train_df = df.sample(frac=1)
        #test_df = t_df.sample(frac=.2)
        
        # print("\nDimensions\n")
        
        # model.neuralNet.printDim(4)
        
        # print("\weight\n")
        # model.neuralNet.printWeights(4)
        
        # print("\bias\n")
        # model.neuralNet.printBias(4)
        
        
        truth_subset = train_df.iloc[:,0].to_numpy()
        train_subset = train_df.iloc[:,1:].to_numpy()/255
        encoded_truth = oneHotEncode(truth_subset,10)
        
        
        max_examples = encoded_truth.shape[0]
        #print("Max Sample size: " + str(max_examples))
        #print("Max training: " + str(train_subset.shape[0]))
        #print(train_subset)
        #print(encoded_truth)
        index = 0
        avgLoss = 0
        #print("ready for loop")
        while index < max_examples:
            #print(index)

            model.neuralNet.zeroGrad()
            
            # print("Weight Gradients")
            # model.neuralNet.printGrad()
            
            #print("After grad")
            counter = 0
            while counter < batch_size:
                if index >= max_examples:
                    break
                #print(counter)
                model.neuralNet.setInput(train_subset[index])
                predictions = model.forward(train_subset[index])
                #print(predictions)
                #print("Encoded Truth: " + str(encoded_truth[index]))
                loss = model.neuralNet.crossEntropy(predictions, encoded_truth[index])
                avgLoss += loss
                model.neuralNet.backwardStep(predictions,encoded_truth[index])
                index+=1
                counter+=1
                
                # print("Weight Gradients")
                # model.neuralNet.printGrad()
                
                # print("Bias Gradients")
                # model.neuralNet.printBiasGrad()
            #print("finished batch")
            model.neuralNet.updateWeights()
        print("Avg Loss: " + str(avgLoss/max_examples) + " for epoch: " + str(e))
        #testAccuracy(model,test_df)
    print("finished")
    exit(1)
            
        
        
        
    
    

if __name__ == "__main__":
    
    learningRate = 0.002
    numThreads = 8
    epoch = 50
    batch = 200
    
    trainingDataPath = "../data/mnist_train.csv"
    testingDataPath = "../data/mnist_test.csv"
    
    model = modelMLP()
    
    main(learningRate, numThreads, epoch, batch, trainingDataPath, testingDataPath, model)
    
