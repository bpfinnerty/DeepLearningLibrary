
import pandas as pd
import projNet
import numpy as np
from toyModel import *


def main(learningRate, numThreads, epoch, batch_size, model):
    
    model.neuralNet.setLearningRate(learningRate)
    projNet.setThreads(numThreads)
    
    initialWeights = np.array([[-2.5,-1.5,.6,.4],[-.1,2.4,-2.2,1.5,-5.2,3.7]],dtype=object)
    initialBias = np.array([[1.6,.7],[-2,0,1]],dtype=object)
    target = [1,0,0]
    
    for i in range(initialWeights.shape[0]):
        print("\nPre Layer "+str(i)+" Weights & Bias:\n")
        model.neuralNet.printWeights(i)
        model.neuralNet.printBias(i)
    
    for i in range(initialWeights.shape[0]):
        
        print("\nPre Layer "+str(i)+" Weights & Bias:\n")
        model.neuralNet.setWeights(initialWeights[i],i)
        model.neuralNet.setBias(initialBias[i],i)
        
    for i in range(initialWeights.shape[0]):
        print("\nLayer "+str(i)+" Weights & Bias:\n")
        model.neuralNet.printWeights(i)
        model.neuralNet.printBias(i)
    
    inputArray = np.array([.04,.42])
    
    softMaxPred = model.forward(inputArray)
    print("Soft max output")
    print(softMaxPred)
    
    print("Backward Step")
    model.neuralNet.backwardStep(softMaxPred,target)
    
    print("Weight Gradients")
    model.neuralNetwork.printGrad()
    
    print("Bias Gradients")
    model.neuralNetwork.printBiasGrad()
    
    
        
            
        
        
        
    
    

if __name__ == "__main__":
    
    learningRate = 0.01
    numThreads = 1
    epoch = 1
    batch = 1
    
    model = modelMLP()
    
    main(learningRate, numThreads, epoch, batch, model)