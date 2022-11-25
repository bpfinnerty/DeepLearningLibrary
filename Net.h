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

class Net{
    
    public:
        struct Layer{
            double* weights;
            double* bias;
            double* nodeOutput;
            double* deltaList;
            double* deltaBias;
            int* inputs;
            int* outputs;
        };


        // Layer(int input, int output): inputs(input), outputs(output) {
        //     weights.resize(inputs*outputs);
        //     bias.resize(outputs);
        //     nodeOutput.resize(outputs);
            
        //     normal_distribution_weights();
            
        //     deltaList.resize(inputs*outputs,0.0);
        //     deltaBias.resize(outputs,0.0);
        // }

        void normal_distribution_weights(double* weights, double* bias, int inputs, int outputs);
        void setWeights(std::vector<double> startWeights, int layer);
        void setBias(std::vector<double> startBias, int layer);

        std::vector<double> ff(std::vector<double> x, int layer);
        int getInputs();
        int getOutputs();



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
        std::vector<double> Sigmoid(std::vector<double> x);
        double sigmoid_deriv(double x);

        void printBias(int layer);
        void printGrad();
        
        //double MSLOSS(std::vector<double> inputs,std::vector<double> target);
        double crossEntropy(std::vector<double> output,std::vector<double> target);
        //std::vector<double> crossEntropy_deriv(std::vector<double> output,std::vector<double> target);
        std::vector<double> softMax(std::vector<double> x);
        double softMax_deriv(double* x, int focus, int outputLen);

        
        //std::vector<double> dotproduct(std::vector<double> inputs);
        
        void backwardStep(std::vector<double> output,std::vector<double> target);
        void updateWeights();
        void zeroGrad();

        void printWeights(int layer);
        void printDim(int layer);

        bool gpu_check = false;
        bool softBool = false;
        bool meanLoss = false;
        int totalTrain = 0;

        Net(){

        }
};



#endif