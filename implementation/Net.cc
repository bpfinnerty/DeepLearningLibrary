#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include <cmath>
#include<vector>
#include<math.h>
#include<stdexcept>
#include<iostream>
#include<fstream>
#include<boost/algorithm/string.hpp>
#include<boost/lexical_cast.hpp>
#include "Net.h"

//namespace Config{
//    void setThreads(int t){
//        numThreads = t;
//    }
//}

void Net::setThreadNum(int num){
    numOfThreads = num;
}

void Net::setInput(std::vector<double> x){
    int s = firstInputs.size();
    int xSize = x.size();
    if(s==0){
        for(int i = 0;i<xSize ;++i){
            firstInputs.push_back(x[i]);
        }
    }else{
        for(int i = 0;i<xSize ;++i){
            firstInputs[i] = x[i];
        }
    }
}

void Net::addLayer(int inputs, int outputs){

    std::vector<double> weights(inputs*outputs);
    std::vector<double> bias(outputs);
    std::vector<double> nodeOutput(outputs);
    
    normal_distribution_weights(weights.data(),bias.data(),inputs,outputs);
    
    std::vector<double> deltaList(inputs*outputs,0.0);
    std::vector<double> deltaBias(outputs,0.0);


    net.push_back(weights);
    net.push_back(bias);
    net.push_back(nodeOutput);
    net.push_back(deltaList);
    net.push_back(deltaBias);

    sizes.push_back(inputs);
    sizes.push_back(outputs);
}

void Net::setWeights(std::vector<double> startWeights, int layer){
    if(layer>=(int)(net.size()/layerSize)){
        throw std::runtime_error("Layer out of bounds");
    }
    
    double* weights = net[layer*layerSize+ weightsOffset].data();
    int inputs = sizes[layer*2];
    int outputs = sizes[layer*2+1];

    if((int)startWeights.size() > inputs*outputs){
        throw std::runtime_error("Mismatched dimensions for initialize weights");
    }
    for(int i = 0; i<(int)startWeights.size();++i){
        weights[i] = startWeights[i];
    }
    // for(int i = 0; i<(int)startWeights.size();++i){
    //     std::cout << weights[i] << " ,";
    // }
    // std::cout << "\n";
}


void Net::setBias(std::vector<double> startBias, int layer){
    if(layer>=(int)(net.size()/layerSize)){
        throw std::runtime_error("Layer out of bounds");
    }
    double* bias = net[layer*layerSize+ biasOffset].data();
    int inputs = sizes[layer*2];
    int outputs = sizes[layer*2+1];

    if((int)startBias.size() > inputs*outputs){
        throw std::runtime_error("Mismatched dimensions for initialize bias");
    }
    for(int i = 0; i<(int)startBias.size();++i){
        bias[i] = startBias[i];
        // std::cout << bias[i] << " ,";
    }
    // for(int i = 0; i<(int)startBias.size();++i){
    //     std::cout << bias[i] << " ,";
    // }
    // std::cout << "\n";
}

void Net::printDim(int layer){
    int inputs = sizes[layer*2];
    int outputs = sizes[layer*2+1];
    std::cout << "Inputs: " << inputs << " Outputs: " << outputs << "\n";
}


void Net::setLearningRate(double x){
    learningRate = x;
}

std::vector<double> Net::relu(std::vector<double> x){
    //std::cout << "Relu Time\n";
    int size = x.size();
    if((int)activations.size() != (int)(net.size()/layerSize)){
        // td::cout << "addToActivations\n";
        activations.push_back(&Net::relu);
        activations_deriv.push_back(&Net::relu_deriv);
    }
    std::vector<double> ret(size);
    
    #pragma omp parallel for num_threads(numOfThreads)
    for(int i = 0; i<size;++i){
        ret[i] = x[i] > 0 ? x[i] : 0;
    }
    
    return ret;
}

std::vector<double> Net::leakyRelu(std::vector<double> x){
    //std::cout << "Time for leaky Relu\n";
    
    if((int)activations.size() != (int)(net.size()/layerSize)){
        //std::cout << "addToActivations\n";
        activations.push_back(&Net::leakyRelu);
        activations_deriv.push_back(&Net::leakyRelu_deriv);
    }
    
    int size = x.size();
    std::vector<double> ret(size);
    
    #pragma omp parallel for num_threads(numOfThreads)
    for(int i = 0; i<size;++i){
        ret[i] = x[i] > 0 ? x[i] : x[i]*0.01;
    }
    
    return ret;
}

double Net::msLoss(std::vector<double> x,std::vector<double> target){
    if(x.size() != target.size()){
        throw std::runtime_error("Mismatched dimensions for Loss");
    }
    
    meanLoss = true;
    double ret = 0.0;
    int size = x.size();
    for(int i = 0; i<size;++i){
        ret += (pow((x[i]-target[i]),2));
    }
    ret = ret*.5;
    return ret;
}

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
    //std::cout << "Soft max Time\n";
    if((int)activations.size() != (int)(net.size()/layerSize)){
        // std::cout << "addToActivations\n";
        activations.push_back(&Net::softMax);
        activations_deriv.push_back(&Net::leakyRelu_deriv);
    }
    //std::cout << "hitLimit " << ((int)activations.size() != (int)(net.size()/layerSize)) << "\n";
    softBool = true;
    std::vector<double> ret((int)x.size(),0.0);
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
    #pragma omp parallel for num_threads(numOfThreads)
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
    int totalLayer = net.size()/layerSize;
    //std::cout << "Numer of Layers: " << totalLayer << "\n";
    
    for(int layer = 0; layer<totalLayer;++layer){
        //std::cout << "Layer: " << layer;
        int numInputs = sizes[layer*2];
        int numOutputs = sizes[layer*2+1];

        //std::cout << "Inputs: " << numInputs << " Outputs: " << numOutputs << "\n";
        double* d = net[layer*layerSize+ weightDeltaOffset].data();
        double* db = net[layer*layerSize+ biasDeltaOffset].data();
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
    
    if(firstInputs.size() == 0){
        throw std::runtime_error("Must set function input with setInput()\n");
    }
    double* firstInputLayer = firstInputs.data();
    std::vector<double> errors;
    totalTrain++;
    int targetSize = target.size();

    int layers = net.size()/layerSize;
    for(int curLayer = layers-1; curLayer>=0; --curLayer){
        
        int numInputs = sizes[curLayer*2];
        int numOutputs = sizes[curLayer*2+1];
        
        double* weights = net[curLayer*layerSize+ weightsOffset].data();
        double* nodeOutput = net[curLayer*layerSize+ outputOffset].data();
        double* deltaBias = net[curLayer*layerSize+ biasDeltaOffset].data();
        double* deltaList = net[curLayer*layerSize+ weightDeltaOffset].data();

        if(curLayer == layers-1 && layers-1 != 0){
            errors.resize(targetSize);

            std::vector<double> act = net[(curLayer-1)*layerSize+ outputOffset];
            std::vector<double> dataList;
            double* inputList;
            
            // get errors output node
            if(meanLoss){
                #pragma omp parallel for num_threads(numOfThreads)
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
                //std::cout << "usingSoftBool\n";
                dataList = softMax(act);
                inputList = dataList.data();
                for(int i = 0; i<numOutputs;++i){
                    // error * derivative of activation
                    errors[i] *= softMax_deriv(nodeOutput,i,numOutputs);
                }
            }else{
                auto fp = activations[curLayer-1];
               
                dataList = (this->*fp)(act);
               
                inputList = dataList.data();
               

               
                auto funcPointer = activations_deriv[curLayer];
                #pragma omp parallel for num_threads(numOfThreads)
                for(int i = 0; i<numOutputs;++i){
                    double outputActivation = (this->*funcPointer)(nodeOutput[i]);
                    errors[i] *= outputActivation;
                }
               
            }

            #pragma omp parallel for num_threads(numOfThreads)
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
            auto funcPointer = activations_deriv[curLayer];
            #pragma omp parallel for num_threads(numOfThreads)
            for(int i = 0; i<numOutputs; ++i){
                // error * derivative of activation
                double deriv_temp = (this->*funcPointer)(nodeOutput[i]);
                errors[i] *= deriv_temp;
                
                // get gradient for delta term.
                deltaBias[i] += errors[i];
                
                // get gradients for all weights
                for(int j = 0; j<numInputs; ++j){
                    deltaList[j*numOutputs+i] += errors[i] * firstInputLayer[j];
                }
            }
        }
        else{
            std::vector<double> act = net[(curLayer-1)*layerSize+ outputOffset];
            std::vector<double> dataList;
            double* inputList;

            auto fp = activations[curLayer-1];
            dataList = (this->*fp)(act);
            inputList = dataList.data();
           
            // in order to update, still need deriv of activation and previous input. That will give delta. w = w - learning rate*delta PLUM
            auto funcPointer = activations_deriv[curLayer]; 
            #pragma omp parallel for num_threads(numOfThreads)
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
                newErrors[i] += errors[j]*weights[i*numOutputs+j];
            }
            //std::cout << "New Errors " << i << " " << newErrors[i] << "\n";
        }
        errors = newErrors;        
    }  
    firstInputs.clear();
}

void Net::printWeights(int layer){
    int input = sizes[layer*2];
    int output = sizes[layer*2+1];
    
    double* weights = net[layer*layerSize+ weightsOffset].data();
        
    std::cout << "[";
    for(int i = 0; i< input*output;++i){
        std::cout << weights[i] << " ,";
    }
    std::cout << "]\n";
}

void Net::writeWeightsToFile(){
    int num_of_layers = net.size()/layerSize;

    std::ofstream file;
    file.open("net_weights_bias.txt", std::ios_base::out);

    // Iterate over number of layers 
    for(int i = 0; i < num_of_layers; i++){
        // for each layer, grab size of input and output
        int input = sizes[i * 2];
        int output = sizes[i * 2 + 1];

        // retrieve the weights and bias for each layer
        double* weights = net[i * layerSize + weightsOffset].data();
        double* bias = net[i * layerSize + biasOffset].data();

        // add a line that contains the layer number, input and output
        file << i << "," << input << "," << output << "\n";

        // on the next line, add all of the weights to the file
        for(int j = 0; j < input * output; j++){
            file << weights[j];
            if(j + 1 != input * output){
                file << ",";
            }
        }
        file << "\n";

        // on the next line, add all of the bias to the file
        for(int j = 0; j < output; j++){
            file << bias[j];
            if(j + 1 != output){
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

void Net::readWeightsFromFile(std::string filename){
    std::ifstream file(filename);
    std::string line;

    int layer = -1;
    int input = 0;
    int output = 0;
    bool weightsProcessed = false;
    bool biasProcessed = false;

    while(getline(file, line)){
        std::vector<std::string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(","));

        if(layer == -1 && input == 0 && output == 0){
            try{
                //std::cout << vec.at(0) << " " << vec.at(1) << " " << vec.at(2) << "\n";
                layer = boost::lexical_cast<int>(vec.at(0));
                input = boost::lexical_cast<int>(vec.at(1));
                output = boost::lexical_cast<int>(vec.at(2));
            }catch(boost::bad_lexical_cast &e){
                std::cout << "Could not parse layer number and layer input and output sizes\n";
                layer = -1;
                input = 0;
                output = 0;
                continue;
            }
        }else{
            if(!weightsProcessed){
                double* weights = net[layer * layerSize + weightsOffset].data();
                for(int i = 0; i < (int)vec.size(); i++){
                    try{
                        double val = boost::lexical_cast<double>(vec.at(i));
                        weights[i] = val;
                    }catch(boost::bad_lexical_cast &e){
                        std::cout << " unable to parse value from file, filling with zero\n";
                    }
                }
                weightsProcessed = true;
                continue;
            }
            if(!biasProcessed){
                double* bias = net[layer * layerSize + biasOffset].data();
                for(int i = 0; i < (int)vec.size(); i++){
                    try{
                        double val = boost::lexical_cast<double>(vec.at(i));
                        bias[i] = val;
                    }catch(boost::bad_lexical_cast &e){
                        std::cout << " unable to parse value from file, filling with zero\n";
                    }
                }
                layer = -1;
                input = 0;
                output = 0;
                weightsProcessed = false;
                biasProcessed = false;
                continue;
            }
        }
    }
    file.close();
}

void Net::printBias(int layer){
    int output = sizes[layer*2+1];
    double* bias = net[layer*layerSize+ biasOffset].data();
    std::cout << "[";
    for(int i = 0; i< output;++i){
        std::cout << bias[i] << " ,";
    }
    std::cout << "]\n";
}

void Net::printGrad(){
   
    for(int i = 0; i<(int)net.size()/layerSize;++i){
        std::cout << "\nLayer: " << i << "\n";
        int inputs = sizes[i*2];
        int outputs = sizes[i*2+1];
        double* deltaList = net[i*layerSize+ weightDeltaOffset].data();
        std::cout << "[";
        for(int j = 0; j<inputs*outputs;++j){
            std::cout << deltaList[j] << " ,";
        }
        std::cout << "]\n";
    }
}

void Net::printBiasGrad(){
    for(int i = 0; i<(int)net.size()/layerSize;++i){
        
        std::cout << "\nLayer: " << i << "\n";
        int outputs = sizes[i*2+1];
        double* biasDelta = net[i*layerSize+ biasDeltaOffset].data();
        std::cout << "[";
        for(int j = 0; j<outputs;++j){
            std::cout << biasDelta[j] << " ,";
        }
        std::cout << "]\n";
    }
}

void Net::updateWeights(){
    int layers = net.size()/layerSize;
    // update gradients
    for(int currLayer = 0;currLayer < layers;++currLayer){
        int numInputs = sizes[currLayer*2];
        int numOutputs = sizes[currLayer*2+1];

        double* weights = net[currLayer*layerSize+ weightsOffset].data();
        double* bias =net[currLayer*layerSize+ biasOffset].data();
        double* deltaBias =net[currLayer*layerSize+ biasDeltaOffset].data();
        double* deltaList = net[currLayer*layerSize+ weightDeltaOffset].data();
        
        #pragma omp parallel for num_threads(numOfThreads)
        for(int j = 0; j < numOutputs; ++j){
            double biasTemp = deltaBias[j];
            biasTemp = biasTemp >= gradMaxThreshold ? gradMaxThreshold : biasTemp;
            biasTemp = biasTemp <= gradMinThreshold ? gradMinThreshold : biasTemp;

            bias[j] -= (learningRate*biasTemp)/totalTrain;
            deltaBias[j] = 0.0;
            for(int i = 0; i<numInputs;++i){
                double temp = deltaList[i*numOutputs+j];
                temp = temp >= gradMaxThreshold ? gradMaxThreshold : temp;
                temp = temp <= gradMinThreshold ? gradMinThreshold : temp;
                weights[i*numOutputs+j] -= (learningRate*temp)/totalTrain;
                deltaList[i*numOutputs+j] = 0.0;
            }
        }
    }
    totalTrain = 0;
}


std::vector<double> Net::Sigmoid(std::vector<double> x){
   
    if((int)activations.size() != (int)(net.size()/layerSize)){
       
        activations.push_back(&Net::Sigmoid);
        activations_deriv.push_back(&Net::sigmoid_deriv);
    }
    std::vector<double> ret((int)x.size(),0.0);
    
    #pragma omp parallel for num_threads(numOfThreads)
    for(int i =0; i<(int)x.size();++i){
        ret[i] = (1.0)/(1.0+exp(-x[i]));

    }
    return ret;
}

double Net::sigmoid_deriv(double x){

    double sigOutput = (1.0)/(1.0+exp(-x));
    return sigOutput * (1-sigOutput);
}

// Methods for Layer Class
void Net::normal_distribution_weights(double* weights, double* bias, int inputs, int outputs){
    int size = inputs*outputs;
    double stdv = 1/sqrt(outputs);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> distribution(stdv,stdv);
    int i = 0;
    
    while(i< size){
        double w = distribution(gen)/1;
        weights[i] = w;
        i++;
    }
    int j =0;
    while(j < outputs){
        bias[j] = 0.0;
        j++;
    }
    //std::f << "Initialized Weights";
}


std::vector<double> Net::ff(std::vector<double> x, int layer){
    if(layer >= (int)net.size()/layerSize){
        throw std::runtime_error("Mismatched dims in index");
    }

    int inputs = sizes[layer*2];
    int outputs = sizes[layer*2+1];

    double* weights = net[layer*layerSize+ weightsOffset].data();
    double* bias =net[layer*layerSize+ biasOffset].data();
    double* nodeOutput = net[layer*layerSize+ outputOffset].data();
        

    if ((int)x.size() != inputs){
        throw std::runtime_error("Mismatched dims in index");
    }
    else{
        //std::cout << "Input is length " << x.size() << "\n";
    }
    //std::cout << "Time for feed forward\n";
    //std::cout << "Outputs: " << outputs << "\n";
    std::vector<double> ret(outputs,0.0);
    double* r = ret.data();
    

    //std::cout << "Set Bias\n";
    #pragma omp parallel for num_threads(numOfThreads)
    for(int j = 0; j<outputs;++j){
        ret[j] = bias[j];
    }

    
    //std::cout << "multiply by weights\n";
    #pragma omp parallel for num_threads(numOfThreads) reduction(+:r[0:outputs])
    for(int i = 0; i < outputs; ++i){
       //std::cout << "pre multiply: " << sum << "\n";
        for(int j = 0; j<inputs; ++j){
            r[i] +=  x[j]*weights[j*outputs+i];
        //std::cout << "Completed weight " << j << "\n";  
        }
        //std::cout << "post multiply: " << sum << "\n";
        nodeOutput[i] = r[i];
        //std::cout << "post Set: " << ret[i] << "\n";
    }
    //std::cout << "finished multiply\n";
    return ret;    
}





