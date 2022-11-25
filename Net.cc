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
#include "Net.h"

namespace Config{
    void setThreads(int t){
        numThreads = t;
    }
}

void Net::addLayer(int inputs, int outputs){
    Layer l;
    std::vector<double> weights(inputs*outputs);
    std::vector<double> bias(outputs);
    std::vector<double> nodeOutput(outputs);

    l.weights = weights.data();
    l.bias = bias.data();
    l.nodeOutput = nodeOutput.data();
    
    normal_distribution_weights(l.weights,l.bias,inputs,outputs);
    
    std::vector<double> deltaList(inputs*outputs,0.0);
    std::vector<double> deltaBias(outputs,0.0);

    l.deltaList = deltaList.data();
    l.deltaBias = deltaBias.data();

    l.inputs = &inputs;
    l.outputs = &outputs;
    
    //std::cout << "Inputs: " << l.inputs << " Outputs: " << l.outputs << "\n";

    net.push_back(&l);
    Layer* test_l = net.back();
    std::cout << "Inputs: " << *(test_l->inputs) << " Outputs: " << *(test_l->outputs) << "\n";
}

void Net::setWeights(std::vector<double> startWeights, int layer){
    if(layer>=(int)net.size()){
        throw std::runtime_error("Layer out of bounds");
    }
    Layer* l = net[layer];
    double* weights = l->weights;
    int inputs = *(l->inputs);
    int outputs = *(l->outputs);
    int total = inputs*outputs;
    if((int)startWeights.size() > total){
        throw std::runtime_error("Mismatched dimensions for initialize weights");
    }
    for(int i = 0; i<(int)startWeights.size();++i){
        weights[i] = startWeights[i];
    }
    for(int i = 0; i<(int)startWeights.size();++i){
        std::cout << weights[i] << " ,";
    }
    std::cout << "\n";
}


void Net::setBias(std::vector<double> startBias, int layer){
    if(layer>=(int)net.size()){
        throw std::runtime_error("Layer out of bounds");
    }
    Layer* l = net[layer];
    double* bias = l->bias;
    int inputs = *(l->inputs);
    int outputs = *(l->outputs);
    int total = inputs*outputs;
    if((int)startBias.size() > total){
        throw std::runtime_error("Mismatched dimensions for initialize bias");
    }
    for(int i = 0; i<(int)startBias.size();++i){
        bias[i] = startBias[i];
        std::cout << bias[i] << " ,";
    }
    for(int i = 0; i<(int)startBias.size();++i){
        std::cout << bias[i] << " ,";
    }
    std::cout << "\n";
}

void Net::printDim(int layer){
    Layer* l = net[layer];
    std::cout << "Inputs: " << l->inputs << " Outputs: " << *(l->outputs) << "\n";
}


void Net::setLearningRate(double x){
    learningRate = x;
}

std::vector<double> Net::relu(std::vector<double> x){
    int size = x.size();
    if(activations.size() != net.size()){
        activations.push_back(&Net::relu_deriv);
    }
    std::vector<double> ret(size);
    
    #pragma omp parallel for num_threads(Config::numThreads)
    for(int i = 0; i<size;++i){
        ret[i] = x[i] > 0 ? x[i] : 0;
    }
    
    return ret;
}

std::vector<double> Net::leakyRelu(std::vector<double> x){
    //std::cout << "Time for leaky Relu\n";
    if(activations.size() != net.size()){
        activations.push_back(&Net::leakyRelu_deriv);
    }
    
    int size = x.size();
    std::vector<double> ret(size);
    
    #pragma omp parallel for num_threads(Config::numThreads)
    for(int i = 0; i<size;++i){
        ret[i] = x[i] > 0 ? x[i] : x[i]*0.01;
    }
    
    return ret;
}

// double MSLOSS(std::vector<double> x,std::vector<double> target){
//     double ret;
//     int size = x.size();

//     return ret;
// }

double Net::relu_deriv(double x){
    if (x <=0){
        return 0.0;
    }
    else{
        return 1.0;
    }
}

double Net::leakyRelu_deriv(double x){
    if (x <=0){
        return 0.01;
    }
    else{
        return 1.0;
    }
}

double Net::crossEntropy(std::vector<double> output,std::vector<double> target){
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

std::vector<double> Net::softMax(std::vector<double> x){
    if(activations.size() != net.size()){
        activations.push_back(&Net::leakyRelu_deriv);
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

double Net::softMax_deriv(double* x, int focus, int outputLen){
    double ret = 0.0;
    #pragma omp parallel for num_threads(Config::numThreads)
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



void Net::zeroGrad(){
    int totalLayer = net.size();
    //std::cout << "Numer of Layers: " << totalLayer << "\n";
    // for(int i = 0; i< totalLayer; ++i){
    //     Layer* l = net[i];
    //     //std::cout << "Inputs: " << l.inputs << " Outputs: " << l.outputs << "\n";
    // }
    for(int layer = 0; layer<totalLayer;++layer){
        //std::cout << "Layer: " << layer;
        Layer* l = net[layer];
        int numInputs = *(l->inputs);
        int numOutputs  = *(l->outputs);

        //std::cout << "Inputs: " << numInputs << " Outputs: " << numOutputs << "\n";
        double* d = l->deltaList;
        double* db = l->deltaBias;
        for(int j = 0; j< numInputs*numOutputs;++j){
            d[j] = 0;
        }
        for(int j = 0; j< numOutputs;++j){
            db[j] = 0;
        }
    }
    //std::cout << "Finised zero grad\n";
    //std::cout << "Does it fail after print\n";
}

void Net::backwardStep(std::vector<double> output,std::vector<double> target){
    std::vector<double> errors;
    
    
    totalTrain++;

    int layers = net.size();
    for(int curLayer = layers-1; curLayer>0; --curLayer){
        Layer* l = net[curLayer];
        int numInputs = *(l->inputs);
        int numOutputs  = *(l->outputs);

        double* weights = l->weights;
        double* nodeOutput = l->nodeOutput;
        double* deltaBias = l->deltaBias;
        double* deltaList = l->deltaList;

        if(curLayer == layers-1 && layers-1 != 0){
            errors.resize(target.size());

            double* inputList = net[curLayer-1]->nodeOutput;
            
            // get errors output node
            if(meanLoss){
            #pragma omp parallel for num_threads(Config::numThreads)
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
                #pragma omp parallel for num_threads(Config::numThreads)
                for(int i = 0; i<numOutputs;++i){
                    errors[i] *= (this->*funcPointer)(nodeOutput[i]);
                }
            }

            #pragma omp parallel for num_threads(Config::numThreads)
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
            #pragma omp parallel for num_threads(Config::numThreads)
            for(int i = 0; i<numOutputs; ++i){
                // error * derivative of activation
                errors[i] *= (this->*funcPointer)(nodeOutput[i]);
                
                // get gradient for delta term.
                deltaBias[i] += errors[i];
                
                // get gradients for all weights
                for(int j = 0; j<numInputs; ++j){
                    deltaList[j*numOutputs+i] += errors[i] * Config::firstInputs[j];
                }
            }
        }
        else{
            double* inputList = net[curLayer-1]->nodeOutput;
            // in order to update, still need deriv of activation and previous input. That will give delta. w = w - learning rate*delta
            auto funcPointer = activations[curLayer];
            #pragma omp parallel for num_threads(Config::numThreads)
            for(int i = 0; i<numOutputs; ++i){
                // error * derivative of activation
                errors[i] *= (this->*funcPointer)(nodeOutput[i]);
                
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
            errors[i] = newErrors[i];
        }
    }  
    Config::firstInputs = NULL;
}

void Net::printWeights(int layer){
    std::cout << "In Print Weights";
    Layer* l = net[layer];
    int input = *(l->inputs);
    int output = *(l->outputs);
    std::cout << "Inputs: " << *(l->inputs) << " Outputs: " << *(l->outputs) << "\n";
    double* weights = l->weights;
    std::cout << "[";
    for(int i = 0; i< input*output;++i){
        std::cout << weights[i] << " ,";
    }
    std::cout << "]\n";
}

void Net::printBias(int layer){
    Layer* l = net[layer];
    int output = *(l->outputs);
    double* bias = l->bias;
    std::cout << "[";
    for(int i = 0; i< output;++i){
        std::cout << bias[i] << " ,";
    }
    std::cout << "]\n";
}

void Net::printGrad(){
    
    for(int i = 0; i<(int)net.size();++i){
        Layer* l = net[i];
        int inputs = *(l->inputs);
        int outputs = *(l->outputs);
        int total = inputs*outputs;
        std::cout << "[";
        for(int j = 0; j< total;++j){
            std::cout << l->deltaList[j] << " ,";
        }
        std::cout << "]\n";
    }
}

void Net::updateWeights(){
    int layers = net.size();
    // update gradients
    for(int currLayer = 0;currLayer < layers;++currLayer){
        int numInputs = *(net[currLayer]->inputs);
        int numOutputs  = *(net[currLayer]->outputs);

        double* weights = net[currLayer]->weights;
        double* bias = net[currLayer]->bias;
        double* deltaBias = net[currLayer]->deltaBias;
        double* deltaList =net[currLayer]->deltaList;
        
        #pragma omp parallel for num_threads(Config::numThreads)
        for(int j = 0; j < numOutputs; ++j){
            bias[j] -= (learningRate*deltaBias[j])/totalTrain;
            deltaBias[j] = 0.0;
            for(int i = 0; i<numInputs;++i){
                weights[i*numOutputs+j] -= (learningRate*deltaList[i*numOutputs+j])/totalTrain;
                deltaList[i*numOutputs+j] = 0.0;
            }
        }
    }
    totalTrain = 0;
}


std::vector<double> Net::Sigmoid(std::vector<double> x){
    if(activations.size() != net.size()){
        activations.push_back(&Net::sigmoid_deriv);
    }
    std::vector<double> ret(x.size(),0.0);
    #pragma omp parallel for num_threads(Config::numThreads)
    for(int i =0; i<(int)x.size();++i){
        ret[i] = (1.0)/(1.0+exp(-x[i]));
    }
    return ret;
}

double Net::sigmoid_deriv(double x){
    return x*(1-x);
}

// Methods for Layer Class
void Net::normal_distribution_weights(double* weights, double* bias, int inputs, int outputs){
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> distribution(.05,1.0);
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
    while(j < outputs){
        double w = distribution(gen);
        if (w >= 0.0 && w <=1.0){
            bias[j] = w;
            j++;
        }
    }
    //std::f << "Initialized Weights";
}


std::vector<double> Net::ff(std::vector<double> x, int layer){
    if(layer >= (int)net.size()){
        throw std::runtime_error("Mismatched dims in index");
    }

    Layer* l = net[layer];

    int inputs = *(l->inputs);
    int outputs = *(l->outputs);
    double* bias = l->bias;
    double* weights = l->weights;
    double* nodeOutput = l->nodeOutput;

    if ((int)x.size() != inputs){
        throw std::runtime_error("Mismatched dims in index");
    }
    else{
        //std::cout << "Input is length " << x.size() << "\n";
    }
    //std::cout << "Time for feed forward\n";
    //std::cout << "Outputs: " << outputs << "\n";
    std::vector<double> ret(outputs,0.0);
    
    //std::cout << "Check first input\n";
    if(Config::firstInputs == NULL){
        Config::firstInputs = x.data();
    }

    //std::cout << "Set Bias\n";
    #pragma omp parallel for num_threads(Config::numThreads)
    for(int j = 0; j<outputs;++j){
        ret[j] = bias[j];
    }

    double sum = 0.0;
    //std::cout << "multiply by weights\n";
    #pragma omp parallel for num_threads(Config::numThreads) private(sum)
    for(int i = 0; i < outputs; ++i){
       //std::cout << "Starting Node " << i << "\n";
       sum = ret[i];
       //std::cout << "pre multiply: " << sum << "\n";
        for(int j = 0; j<inputs; ++j){
            sum += x[j]*weights[j*outputs+i];
        //std::cout << "Completed weight " << j << "\n";  
        }
        //std::cout << "post multiply: " << sum << "\n";
        ret[i] = sum;
        nodeOutput[i] = ret[i];
        //std::cout << "post Set: " << ret[i] << "\n";
    }
    //std::cout << "finished multiply\n";
    return ret;    
}




namespace py = pybind11;

PYBIND11_MODULE(projNet,m){  
  py::class_<Net>(m,"Net")
    .def(py::init<>())
    .def("ff",&Net::ff)
    .def("addLayer",&Net::addLayer)
    .def("setLearningRate",&Net::setLearningRate)
    .def("relu",&Net::relu)
    .def("backwardStep",&Net::backwardStep)
    .def("leakyRelu",&Net::leakyRelu)
    .def("updateWeights",&Net::updateWeights)
    .def("crossEntropy",&Net::crossEntropy)
    .def("zeroGrad",&Net::zeroGrad)
    .def("softMax",&Net::softMax)
    .def("printWeights",&Net::printWeights)
    .def("Sigmoid",&Net::Sigmoid)
    .def("printBias",&Net::printBias)
    .def("setWeights",&Net::setWeights)
    .def("setBias",&Net::setBias)
    .def("printDim",&Net::printDim);
    

  m.def("setThreads",&Config::setThreads);
  
}




