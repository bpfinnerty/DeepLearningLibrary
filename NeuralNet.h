#ifndef LAYER_MODEL
#define LAYER_MODEL


#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<vector>
#include <iostream>
#include<math.h>
#include<stdexcept>
#include <random>

namespace Net{

    std::vector<double*> net;
    int netLay = 4;
    std::vector<double (*)(double)> activations;
    std::vector<int> sizes;
    std::vector<double*> delta;
    std::vector<double*> deltaBias;

    double learningRate = .01;
    double* firstInputs;

    void setInputs(double* x);
    void setLearningRate(double x);
    std::vector<double> relu(std::vector<double> x);
    double relu_deriv(double x);
    std::vector<double> leakyRelu(std::vector<double> x);
    double leakyRelu_deriv(double x);
    
    //double MSLOSS(std::vector<double> inputs,std::vector<double> target);
    //std::vector<double> Cross_Entropy(std::vector<double> inputs,std::vector<double> target);
    
    //std::vector<double> dotproduct(std::vector<double> inputs);
    
    void backwardStep(std::vector<double> x);
    bool gpu_check = false;
}

class NeuralNet{
    
    public:
        std::vector<double> weights;
        std::vector<double> bias;
        std::vector<double> nodeOutput;
        std::vector<double> deltaList;
        std::vector<double> deltaBias;
        int inputs;
        int outputs;


        NeuralNet(int input, int output): inputs(input), outputs(output) {
            Net::sizes.push_back(inputs);
            Net::sizes.push_back(outputs);

            weights.resize(inputs*outputs);
            bias.resize(outputs);
            nodeOutput.resize(outputs);
            
            normal_distribution_weights();
            
            deltaList.resize(inputs*outputs);
            deltaBias.resize(outputs,0.0);
            Net::delta.push_back(deltaList.data());

            Net::net.push_back(weights.data());
            Net::net.push_back(bias.data());
            Net::net.push_back(nodeOutput.data());
            Net::net.push_back(deltaBias.data());

        }

        void normal_distribution_weights();

        std::vector<double> ff(std::vector<double> x);
};

#endif