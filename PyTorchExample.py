import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as f
import torch.utils.data as data_utils
from PyTorchDefinition import *
import os

def main(learningRate, numThreads, epoch, batch_size, trainDataPath,testDataPath, model):
    
    train_df = pd.read_csv(trainDataPath,header=None,skiprows=1)
    test_df = pd.read_csv(testDataPath,header=None,skiprows=1)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    criterion = nn.CrossEntropyLoss()
    
    trainLabelNumpy = train_df.iloc[:,0].to_numpy()
    trainDataNumpy = train_df.iloc[:,1:].to_numpy()/255
    
    target = torch.tensor(trainLabelNumpy)
    features = torch.tensor(trainDataNumpy)
    torch.set_num_threads(8)
    train = data_utils.TensorDataset(features, target)
    train_loader = data_utils.DataLoader(train, batch_size=1, shuffle=True)
    
    testLabelNumpy = test_df.iloc[:,0].to_numpy()
    testDataNumpy = test_df.iloc[:,1:].to_numpy()/255
    
    target_test = torch.tensor(testLabelNumpy)
    test_features = torch.tensor(testDataNumpy)
    
    test = data_utils.TensorDataset(test_features, target_test,)
    test_loader = data_utils.DataLoader(test, batch_size=1, shuffle=True)
    
    for e in range(epoch):
        model.train(True)
        running_loss = 0.0        
        for i, data in enumerate(train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs = inputs.to(torch.float32)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = criterion(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
        
        
        model.train(False)
        accuracy = 0
        
        for i, tdata in enumerate(test_loader):
            inputs,labels = tdata
            inputs = inputs.to(torch.float32)
            output = model(inputs)
            pred = torch.argmax(output)
            if pred == labels:
                accuracy +=1
                
        print("Epoch: {0} Avg Loss: {1} Accuracy: {2}".format(e,running_loss/len(train_loader),accuracy/len(test_loader)))
            

if __name__ == "__main__":
    
    learningRate = 0.04
    numThreads = 8
    epoch = 10
    batch = 100
    
    trainingDataPath = "DeepLearningLibrary/mnist_train.csv"
    testingDataPath = "DeepLearningLibrary/mnist_test.csv"
    
    model = MLP()
    
    main(learningRate, numThreads, epoch, batch, trainingDataPath, testingDataPath, model)