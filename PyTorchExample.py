import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as f
import torch.utils.data as data_utils
from PyTorchDefinition import *

def main(learningRate, numThreads, epoch, batch_size, trainDataPath,testDataPath, model):
    
    train_df = pd.read_csv(trainDataPath,header=None,skiprows=1)
    test_df = pd.read_csv(testDataPath,header=None,skiprows=1)
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    criterion = nn.CrossEntropyLoss()
    
    target = torch.tensor(train_df.iloc[:,0].values)
    features = torch.tensor(train_df.iloc[:,1:].values)

    train = data_utils.TensorDataset(features, target)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    
    target_test = torch.tensor(test_df.iloc[:,0].values)
    test_features = torch.tensor(test_df.iloc[:,1:].values)
    
    test = data_utils.TensorDataset(test_features, target_test)
    test_loader = data_utils.DataLoader(test, batch_size=1, shuffle=True)
    
    
    
    for e in range(epoch):
        model.train(True)
        running_loss = 0.0
        last_loss = 0.0
        
        for i, data in enumerate(train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

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
            if i % batch_size ==batch_size-1:
                last_loss = running_loss / batch_size # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = e * len(train_loader) + i + 1
                print("Loss/Train: " + str(last_loss) + ",  "+ str(tb_x))
                running_loss = 0.0
        
        model.train(False)
        accuracy = 0
        counter = 0
        for i, tdata in enumerate(test_loader):
            inputs,labels = tdata
            output = model(inputs)
            pred = torch.max(output)
            if pred == labels:
                accuracy +=1
            counter += 1
        print("Accuracy for epoch  " + str(e) + " = " +  str(accuracy/counter))
            
            

        
        
        
    
    

if __name__ == "__main__":
    
    learningRate = 0.02
    numThreads = 12
    epoch = 3
    batch = 100
    
    trainingDataPath = "mnist_train.csv"
    testingDataPath = "mnist_test.csv"
    
    model = MLP()
    
    main(learningRate, numThreads, epoch, batch, trainingDataPath, testingDataPath, model)