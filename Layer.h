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

#endif