
import pandas as pd
import projNet
import numpy as np
from toyModel import *


def main(learningRate, numThreads, epoch, batch_size, model):
    
    model.neuralNet.setLearningRate(learningRate)
    projNet.setThreads(numThreads)
    
    initialWeights = np.array([[.15,.25,.20,.30],[.40,.70,.45,.55]],dtype=object)
    initialBias = np.array([[.35,.35],[.60,.60]],dtype=object)
    target = [.01,.99]
    
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
    
    inputArray = np.array([.05,.10])
    
    softMaxPred = model.forward(inputArray)
    print("Soft max output")
    print(softMaxPred)
    
    print("Entropy Loss")
    loss = model.neuralNet.MSLOSS(softMaxPred,target)
    print(loss)
    
    print("Backward Step")
    model.neuralNet.backwardStep(softMaxPred,target)
    
    print("Weight Gradients")
    model.neuralNet.printGrad()
    
    print("Bias Gradients")
    model.neuralNet.printBiasGrad()
    
    
    print("test Cross Entropy")
    print(model.neuralNet.crossEntropy(np.array([.7,.2,.1]),np.array([1,0,0])))
    
    
        
            
        
        
        
    
    

if __name__ == "__main__":
    
    learningRate = 0.01
    numThreads = 1
    epoch = 1
    batch = 1
    
    model = modelMLP()
    
    main(learningRate, numThreads, epoch, batch, model)