

#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<vector>
#include<math.h>
#include<stdexcept>
#include "NeuralNet.h"


namespace NeuralNet{
    std::vector<double> relu(std::vector<double> inputs){
        int size = inputs.size();
        std::vector<double> ret(size);
        if(gpu_check){

        }
        else{
            #pragma omp parallel for
            for(int i = 0; i<size;++i){
                ret[i] = inputs[i] > 0 ? inputs[i] : 0;
            }
        }
        return ret;
    }

    std::vector<double> leakyRelu(std::vector<double> inputs){
        int size = inputs.size();
        std::vector<double> ret(size);
        if(gpu_check){

        }
        else{
            #pragma omp parallel for
            for(int i = 0; i<size;++i){
                ret[i] = inputs[i] > 0 ? inputs[i] : inputs[i]*0.01;
            }
        }
        return ret;
    }

    double MSLOSS(std::vector<double> outputs,std::vector<double> target){
        double ret;
        int size = outputs.size();

        for(int i = 0; i < size; ++i){
            ret =  
        }
    }

    std::vector<double> FeedForward::ff(std::vector<double> x){
        if (x.size() != inputs){
            throw std::runtime_error("Mismatched dims in index");
        }
        std::vector<double> ret(outputs);
        if(gpu_check){

        }
        else{
            #pragma omp parallel for reduction(+:ret[0:outputs])
            for(int i = 0; i < inputs; ++i){
                for(int j = 0; j<outputs; ++j){
                    ret[i] += x[i]*weights[i*outputs+j];  
                }
            }
            return ret;
        }
    }


}



