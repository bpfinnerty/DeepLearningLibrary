#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include <cmath>
#include<vector>
#include<math.h>
#include<stdexcept>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include "NeuralNet.h"


namespace Net{
    // All useful functions

    void setInputs(double* x){
        firstInputs = x;
    }

    void setThreads(int t){
        numThreads = t;
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
            #pragma omp parallel for num_threads(numThreads)
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
        std::cout << "Time for leaky Relu";
        int size = x.size();
        std::vector<double> ret(size);
        if(gpu_check){

        }
        else{
            #pragma omp parallel for num_threads(numThreads)
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

    double crossEntropy(std::vector<double> output,std::vector<double> target){
        double loss = 0.0;
        if(output.size() != target.size()){
            throw std::runtime_error("Mismatched dimensions for Loss");
        }
        int s = output.size();
        for(int i = 0; i<s;i++){
            loss += target[i]*log2(output[i]);
        }

        return loss*-1;
    }

    std::vector<double> softMax(std::vector<double> x){
        if(activations.size() != net.size()/netLay){
            activations.push_back(&leakyRelu_deriv);
        }
        softBool = true;
        std::vector<double> ret(x.size(),0.0);
        double total = 0.0;
        int s = x.size();
        for(int i = 0; i<s;++i){
            total += exp(x[i]);
        }
        for(int i = 0; i<s;++i){
            ret[i] = exp(x[i])/total;
        }

        return ret;
    }

    double softMax_deriv(double* x, int focus, int outputLen){
        double ret = 0.0;
        #pragma omp parallel for num_threads(numThreads)
        for(int i = 0; i<outputLen;++i){
            if(i == focus){
                ret += x[focus] * (1-x[focus]);
            }
            else{
                ret += -x[focus] * x[i];
            }
        }

        return ret;
    }



    void zeroGrad(){
        int totalLayer = delta.size();
        std::cout << "Numer of Layers: " << totalLayer << "\n";
        for(int layer = 0; layer<totalLayer;++layer){
            std::cout << "Layer: " << layer;
            int i = sizes[layer];
            int o = sizes[layer];
            std::cout << "Inputs: " << i << " Outputs: " << o << "\n";
            double* d = delta[layer];
            double* db = net[layer*netLay+3];
            for(int j = 0; j< o*i;++j){
                d[j] = 0;
            }
            for(int j = 0; j< o;++j){
                db[j] = 0;
            }
        }
        std::cout << "Finised zero grad\n";
        std::cout << "Does it fail after print\n";
    }

    void backwardStep(std::vector<double> output,std::vector<double> target){
        std::vector<double> errors;
        if(Net::gpu_check){

        }
        else{
            totalTrain++;

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
                    if(meanLoss){
                    #pragma omp parallel for num_threads(numThreads)
                        for(int i = 0; i <numOutputs; ++i){
                            errors[i]=output[i]-target[i];
                        }
                    }else{
                        for(int i = 0; i < numOutputs;++i){
                            errors[i]=-1.0/output[i];
                        }
                    }

                    // in order to update, still need deriv of activation and previous input. That will give delta. w = w - learning rate*delta
                    if(softBool){
                        for(int i = 0; i<numOutputs;++i){
                            // error * derivative of activation
                            errors[i] *= softMax_deriv(nodeOutput,i,numOutputs);
                        }
                    }else{
                        
                        auto funcPointer = activations[curLayer];
                        #pragma omp parallel for num_threads(numThreads)
                        for(int i = 0; i<numOutputs;++i){
                            errors[i] *= funcPointer(nodeOutput[i]);
                        }
                    }

                    #pragma omp parallel for num_threads(numThreads)
                    for(int i = 0; i<numOutputs; ++i){
                        
                        
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
                    #pragma omp parallel for num_threads(numThreads)
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
                    #pragma omp parallel for num_threads(numThreads)
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
        }
        firstInputs = NULL;
    }

    void updateWeights(){
        int layers = net.size()/netLay;
        // update gradients
        for(int currLayer = 0;currLayer < layers;++currLayer){
            int numInputs = sizes[currLayer*2];
            int numOutputs  =sizes[currLayer*2+1];

            double* weights = net[currLayer*netLay];
            double* bias = net[currLayer*netLay+1];
            double* deltaBias = net[currLayer*netLay+3];
            double* deltaList = delta[currLayer];
            
            #pragma omp parallel for num_threads(numThreads)
            for(int j = 0; j < numOutputs; ++j){
                bias[j] -= learningRate*deltaBias[j];
                deltaBias[j] = 0.0;
                for(int i = 0; i<numInputs;++i){
                    weights[i*numOutputs+j] -= (learningRate*deltaList[i*numOutputs+j])/totalTrain;
                    deltaList[i*numOutputs+j] = 0.0;
                }
            }
        }
        totalTrain = 0;
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
    //std::f << "Initialized Weights";
}


std::vector<double> NeuralNet::ff(std::vector<double> x){
    
    if ((int)x.size() != inputs){
        throw std::runtime_error("Mismatched dims in index");
    }
    std::vector<double> ret(outputs,0.0);
    if(Net::gpu_check){
        return ret;
    }
    else{
        std::cout << "Check first input\n";
        if(Net::firstInputs == NULL){
            Net::firstInputs = x.data();
        }
        std::cout << "Get data\n";
        double* r = ret.data();

        std::cout << "Set Bias\n";
        #pragma omp parallel for num_threads(Net::numThreads)
        for(int j = 0; j<outputs;++j){
            r[j] = bias[j];
        }

        std::cout << "multiply by weights\n";
        #pragma omp parallel for reduction(+:r[0:outputs]) num_threads(Net::numThreads)
        for(int i = 0; i < inputs; ++i){
            for(int j = 0; j<outputs; ++j){
                r[i] += x[i]*weights[i*outputs+j];  
            }
            nodeOutput[i] = r[i];
        }
        return ret;
    }
}

namespace py = pybind11;

PYBIND11_MODULE(projNet,m){
  py::class_<NeuralNet>(m,"NeuralNet")
    .def(py::init<int,int>())
    .def("ff",&NeuralNet::ff);
  
  m.def("setLearningRate",&Net::setLearningRate);
  m.def("setInputs",&Net::setInputs);
  m.def("relu",&Net::relu);
  m.def("backwardStep",&Net::backwardStep);
  m.def("leakyRelu",&Net::leakyRelu);
  m.def("updateWeights",&Net::updateWeights);
  m.def("crossEntropy",&Net::crossEntropy);
  m.def("setThreads",&Net::setThreads);
  m.def("zeroGrad",&Net::zeroGrad);
  
}



