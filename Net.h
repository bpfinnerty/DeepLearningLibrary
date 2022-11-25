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


namespace Config{
    int numThreads = 1;
    void setThreads(int t);
    double* firstInputs = NULL;
}

class Layer{
    
    public:
        std::vector<double> weights;
        std::vector<double> bias;
        std::vector<double> nodeOutput;
        std::vector<double> deltaList;
        std::vector<double> deltaBias;
        int inputs;
        int outputs;


        Layer(int input, int output): inputs(input), outputs(output) {
            weights.resize(inputs*outputs);
            bias.resize(outputs);
            nodeOutput.resize(outputs);
            
            normal_distribution_weights();
            
            deltaList.resize(inputs*outputs,0.0);
            deltaBias.resize(outputs,0.0);
        }

        void normal_distribution_weights();

        std::vector<double> ff(std::vector<double> x);
};

class Net{
    public:
        std::vector<Layer*> net;
        std::vector<double (Net::*)(double)> activations;

        void addLayer(int input, int output);
        Layer getLayer(int layer);

        double learningRate = .01;
        

        void setLearningRate(double x);
        std::vector<double> relu(std::vector<double> x);
        double relu_deriv(double x);
        std::vector<double> leakyRelu(std::vector<double> x);
        double leakyRelu_deriv(double x);
        
        //double MSLOSS(std::vector<double> inputs,std::vector<double> target);
        double crossEntropy(std::vector<double> output,std::vector<double> target);
        //std::vector<double> crossEntropy_deriv(std::vector<double> output,std::vector<double> target);
        std::vector<double> softMax(std::vector<double> x);
        double softMax_deriv(double* x, int focus, int outputLen);

        
        //std::vector<double> dotproduct(std::vector<double> inputs);
        
        void backwardStep(std::vector<double> output,std::vector<double> target);
        void updateWeights();
        void zeroGrad();

        bool gpu_check = false;
        bool softBool = false;
        bool meanLoss = false;
        int totalTrain = 0;

        Net(){

        }
};



#endif