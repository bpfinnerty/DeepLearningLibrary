#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<vector>
#include<math.h>
#include<stdexcept>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include "NeuralNet.h"


namespace NeuralNet{

    // All useful functions

    void setInputs(double* x){
        firstInputs = x;
    }

    void setLearningRate(double x){
        learningRate = x;
    }

    std::vector<double> relu(std::vector<double> x){
        int size = x.size();
        if(activations.size() != net.size()/netLay){
            activations.push_back(&relu_deriv);
        }
        std::vector<double> ret(size);
        if(gpu_check){

        }
        else{
            #pragma omp parallel for
            for(int i = 0; i<size;++i){
                ret[i] = x[i] > 0 ? x[i] : 0;
            }
        }
        return ret;
    }

    std::vector<double> leakyRelu(std::vector<double> x){
        if(activations.size() != net.size()/netLay){
            activations.push_back(&leakyRelu_deriv);
        }
        int size = x.size();
        std::vector<double> ret(size);
        if(gpu_check){

        }
        else{
            #pragma omp parallel for
            for(int i = 0; i<size;++i){
                ret[i] = x[i] > 0 ? x[i] : x[i]*0.01;
            }
        }
        return ret;
    }

    // double MSLOSS(std::vector<double> x,std::vector<double> target){
    //     double ret;
    //     int size = x.size();

    //     return ret;
    // }

    double relu_deriv(double x){
        if (x <=0){
            return 0.0;
        }
        else{
            return 1.0;
        }
    }

    double leakyRelu_deriv(double x){
        if (x <=0){
            return 0.01;
        }
        else{
            return 1.0;
        }
    }

    // std::vector<double> Cross_Entropy(std::vector<double> output,std::vector<double> target){

    // }

    void backward(std::vector<double> output,std::vector<double> target){
        std::vector<double> errors;
        if(gpu_check){

        }
        else{
            

            int layers = net.size()/netLay;
            for(int curLayer = layers-1; curLayer>0; --curLayer){
                
                int numInputs = sizes[curLayer*2];
                int numOutputs  =sizes[curLayer*2+1];

                double* weights = net[curLayer*netLay];
                double* nodeOutput = net[curLayer*netLay+2];
                double* deltaBias = net[curLayer*netLay+3];
                double* deltaList = delta[curLayer];

                if(curLayer == layers-1){
                    errors.resize(target.size());

                    double* inputList = net[(curLayer-1)*netLay+2];
                    
                    // get errors output node
                    #pragma omp parallel for
                    for(int i = 0; i <numOutputs; ++i){
                        errors[i]=output[i]-target[i];
                    }
                    // in order to update, still need deriv of activation and previous input. That will give delta. w = w - learning rate*delta
                    auto funcPointer = activations[curLayer];
                    #pragma omp parallel for
                    for(int i = 0; i<numOutputs; ++i){
                        // error * derivative of activation
                        errors[i] *= (funcPointer(nodeOutput[i]));
                        
                        // get gradient for delta term.
                        deltaBias[i] += errors[i];
                        
                        // get gradients for all weights
                        for(int j = 0; j<numInputs; ++j){
                            deltaList[j*numOutputs+i] += errors[i] * inputList[j];
                        }
                    }
                }
                else if(curLayer == 0){

                    // in order to update, still need deriv of activation and previous input. That will give delta. w = w - learning rate*delta
                    auto funcPointer = activations[curLayer];
                    #pragma omp parallel for
                    for(int i = 0; i<numOutputs; ++i){
                        // error * derivative of activation
                        errors[i] *= (funcPointer(nodeOutput[i]));
                        
                        // get gradient for delta term.
                        deltaBias[i] += errors[i];
                        
                        // get gradients for all weights
                        for(int j = 0; j<numInputs; ++j){
                            deltaList[j*numOutputs+i] += errors[i] * firstInputs[j];
                        }
                    }
                }
                else{
                    double* inputList = net[(curLayer-1)*netLay+2];
                    // in order to update, still need deriv of activation and previous input. That will give delta. w = w - learning rate*delta
                    auto funcPointer = activations[curLayer];
                    #pragma omp parallel for
                    for(int i = 0; i<numOutputs; ++i){
                        // error * derivative of activation
                        errors[i] *= (funcPointer(nodeOutput[i]));
                        
                        // get gradient for delta term.
                        deltaBias[i] += errors[i];
                        
                        // get gradients for all weights
                        for(int j = 0; j<numInputs; ++j){
                            deltaList[j*numOutputs+i] += errors[i] * inputList[j];
                        }
                    }
                }

                std::vector<double> newErrors(numInputs,0.0);
                for(int i = 0; i<numInputs;++i){
                    for(int j = 0; j<numOutputs;++j){
                        newErrors[i] += errors[i]*weights[i*numInputs+j];
                    }
                }
                errors = newErrors;
            }

            // update gradients
            for(int currLayer = 0;currLayer < layers;++currLayer){
                int numInputs = sizes[currLayer*2];
                int numOutputs  =sizes[currLayer*2+1];

                double* weights = net[currLayer*netLay];
                double* bias = net[currLayer*netLay+1];
                double* deltaBias = net[currLayer*netLay+3];
                double* deltaList = delta[currLayer];
                
                #pragma omp parallel for
                for(int j = 0; j < numOutputs; ++j){
                    bias[j] -= learningRate*deltaBias[j];
                    for(int i = 0; i<numInputs;++i){
                        weights[i*numOutputs+j] -= learningRate*deltaList[i*numOutputs+j];
                    }
                }
            }
        }

    }


    // Methods for Layer Class
    void NeuralNet::normal_distribution_weights(){
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
        int j =0;
        while(j < 1){
            double w = distribution(gen);
            if (w >= 0.0 && w <=1.0){
                bias[j] = w;
                j++;
            }
        }
        std::cout << "Initialized Weights";
    }


    std::vector<double> NeuralNet::ff(std::vector<double> x){
        if ((int)x.size() != inputs){
            throw std::runtime_error("Mismatched dims in index");
        }
        if(gpu_check){

        }
        else{
            #pragma omp parallel for
            for(int j = 0; j<outputs;++j){
                nodeOutput[j] = bias[j];
            }

            #pragma omp parallel for reduction(+:nodeOutput[0:outputs])
            for(int i = 0; i < inputs; ++i){
                for(int j = 0; j<outputs; ++j){
                    nodeOutput[i] += x[i]*weights[i*outputs+j];  
                }
            }
            return nodeOutput;
        }
    }


}

namespace py = pybind11;

PYBIND11_MODULE(NeuralNet,m){
  py::class_<NeuralNet>(m,"NeuralNet")
    .def(py::init<int,int>)
    .def("normal_distribution_weights",&NeuralNet::normal_distribution_weights)
    .def("ff",&NeuralNet::ff)
  
  m.def("setLearningRate",&setLearningRate);
  m.def("setInputs",&setInputs);
  m.def("relu",&relu);
  m.def("backward",&backward);
  m.def("leakyRelu",&leakyRelu);
  m.def("matrix_mul_2d",&matrix_mul_2d);
}



