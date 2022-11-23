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

namespace NeuralNet{

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
    std::vector<double> relu(std::vector<double> inputs);
    double MSLOSS(std::vector<double> inputs,std::vector<double> target);
    std::vector<double> Cross_Entropy(std::vector<double> inputs,std::vector<double> target);
    std::vector<double> relu_deriv(std::vector<double> inputs);
    std::vector<double> dotproduct(std::vector<double> inputs);
    void backward(std::vector<double> x);
    bool gpu_check = false;


class NeuralNet{
    
    public:
        std::vector<double> weights;
        std::vector<double> bias;
        std::vector<double> nodeOutput;
        std::vector<double> deltaList;
        std::vector<double> deltaBias;
        char activation;
        int inputs;
        int outputs;


        NeuralNet(int input, int output, char act): inputs(input) outputs(output) activation(act) {
            sizes.push_back(inputs);
            sizes.push_back(outputs);

            weights.resize(inputs*outputs);
            bias.resize(outputs);
            nodeOutput.resize(outputs);
            
            normal_distribution_weights();
            
            deltaList.resize(inputs*outputs);
            deltaBias.resize(outputs,0.0);
            delta.push_back(&deltaList);

            net.push_back(&weights);
            net.push_back(&bias);
            net.push_back(&nodeOutput);
            net.push_back(&deltaBias);

        }

        void normal_distribution_weights();

        std::vector<double> ff(std::vector<double> x);
}

}
#endif