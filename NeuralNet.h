#ifndef LAYER_MODEL
#define LAYER_MODEL


#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<vector>
#include<math.h>
#include<stdexcept>
#include <random>

namespace NeuralNet{

    std::vector<double> relu(std::vector<double> inputs);
    double MSLOSS(std::vector<double> inputs,std::vector<double> target);
    std::vector<double> Cross_Entropy(std::vector<double> inputs,std::vector<double> target);
    std::vector<double> relu_deriv(std::vector<double> inputs);
    std::vector<double> dotproduct(std::vector<double> inputs);
    bool gpu_check = false;

class FeedForward{
    
    public:
        std::vector<double> weights;
        std::vector<double> bias;
        std::vector<double> nodeOutput;
        int inputs;
        int outputs;


        FeedForward(int input, int output){
            inputs = input;
            outputs = output;
            weights.resize(inputs*outputs);
            activation.resize(outputs);

            normal_distribution_weights();

        }

        void normal_distribution_weights(){
            std::random_device rd{};
            std::mt19937 gen{rd()};
            std::normal_distribution<double> distribution(.50,2.0);
            int i = 0;
            int size = inputs*outputs;
            while(i< size){
                double w = distribution(gen);
                if (w >= 0.0 && w <=1.0){
                    weights[i] = w;
                    i++;
                }
            }
            std::cout << "Initialized Weights";
        }

        std::vector<double> ff(std::vector<double> x);
        //std::vector<double> grad(std::vector<double> x);
}

}
#endif