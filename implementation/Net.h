#ifndef NET_MODEL
#define NET_MODEL


#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<vector>
#include <iostream>
#include<math.h>
#include<stdexcept>
#include <random>


class Net{
    
    public:

        int layerSize = 5;
        int weightsOffset = 0;
        int biasOffset = 1;
        int outputOffset = 2;
        int weightDeltaOffset = 3;
        int biasDeltaOffset = 4;
        
        int numOfThreads = 1;

        bool softBool = false;
        bool meanLoss = false;
        int totalTrain = 0;
        double gradMaxThreshold = 10;
        double gradMinThreshold = -10;
        double learningRate = .01;

        void setThreadNum(int num);
        void setInput(std::vector<double> x);
        void normal_distribution_weights(double* weights, double* bias, int inputs, int outputs);
        void setWeights(std::vector<double> startWeights, int layer);
        void setBias(std::vector<double> startBias, int layer);

        std::vector<double> ff(std::vector<double> x, int layer);
        int getInputs();
        int getOutputs();
        std::vector<std::vector<double>> net;
        std::vector<double> firstInputs;
        std::vector<int> sizes;
        std::vector<std::vector<double> (Net::*)(std::vector<double>)> activations;
        std::vector<double (Net::*)(double)> activations_deriv;
        void addLayer(int input, int output);
        void setLearningRate(double x);
        std::vector<double> relu(std::vector<double> x);
        double relu_deriv(double x);
        std::vector<double> leakyRelu(std::vector<double> x);
        double leakyRelu_deriv(double x);
        std::vector<double> Sigmoid(std::vector<double> x);
        double sigmoid_deriv(double x);
        void printBias(int layer);
        void printGrad();
        void printBiasGrad();
        double msLoss(std::vector<double> x,std::vector<double> target);
        double crossEntropy(std::vector<double> output,std::vector<double> target);
        std::vector<double> softMax(std::vector<double> x);
        double softMax_deriv(double* x, int focus, int outputLen);
        void backwardStep(std::vector<double> output,std::vector<double> target);
        void updateWeights();
        void zeroGrad();

        void printWeights(int layer);
        void printDim(int layer);

        void writeWeightsToFile();
        void readWeightsFromFile(std::string filename);

        

        Net(){

        }
};



#endif
