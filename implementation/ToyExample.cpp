#include "Net.h"
#include "data_input.h"
#include <vector>
#include <cstdlib>


std::vector<double> forward(std::vector<double> val, Net* neuralNet){
    std::vector<double> ret = neuralNet->ff(val, 0);
    ret = neuralNet->Sigmoid(ret);
    ret = neuralNet->ff(ret, 1);
    ret = neuralNet->Sigmoid(ret);
    ret = neuralNet->ff(ret, 2);
    // ret = neuralNet.Sigmoid(ret);
    // ret = neuralNet.ff(ret, 3);
    // ret = neuralNet.Sigmoid(ret);
    // ret = neuralNet.ff(ret, 4);
    ret = neuralNet->softMax(ret);
    return ret;
}

std::vector<std::vector<double>> oneHotEncode(std::vector<int> labels, int num_of_classes){
    std::vector<std::vector<double>> zeroed((int)labels.size(), std::vector<double>(num_of_classes, 0));
    for(int i = 0; i < (int)labels.size(); i++){
        int label_val = labels.at(i);
        zeroed[i][label_val] = 1;
    }
    return zeroed;
}

int main(){

    float learningRate = 0.01;
    int numThreads = 8;


    Net model = Net();
    model.addLayer(2, 2);
    model.addLayer(2, 2);
    model.addLayer(2, 3);



    model.setLearningRate(learningRate);
    model.setThreadNum(numThreads);

    std::vector<double> inWeights{-0.3538,0.5640,-0.3729,0.3427};
    std::vector<double> inBias{0.1175,0.6729};

    std::vector<double> h1Weight{-0.6033,-0.4344,-0.2774,0.4700};
    std::vector<double> h1Bias{-0.2599,-0.6711};

    std::vector<double> outputWeight{-0.3809,-0.0079,-0.5754,-0.4448,0.0766,-0.2095};
    std::vector<double> outputBias{0.1251,0.0999,0.4046};

    model.setWeights(inWeights,0);
    model.setWeights(h1Weight,1);
    model.setWeights(outputWeight,2);

    model.setBias(inBias,0);
    model.setBias(h1Bias,1);
    model.setBias(outputBias,2);

    for(int i = 0; i<3;i++){
        std::cout << "Initial Weights\n";
        model.printWeights(i);
        std::cout << "Initial Bias\n\n";
        model.printBias(i);
    }
    
    std::vector<int> truth{0,1,2};
    std::vector<std::vector<double>> train{{.1,.2},{.3,.4},{.6,.7}};
    std::vector<std::vector<double>> encoded_truth = oneHotEncode(truth, 3);
    int max_examples = truth.size();
    int index = 0;
    double avgLoss = 0.0;
    while(index < max_examples){
        model.zeroGrad();
        model.setInput(train[index]);
        std::vector<double> predictions = forward(train[index], &model);
        double loss = model.crossEntropy(predictions, encoded_truth[index]);
        avgLoss += loss;
        model.backwardStep(predictions, encoded_truth[index]);
        index += 1;
        
        
        model.updateWeights();
        for(int i = 0; i<3;i++){
            std::cout << "Current Weights\n";
            model.printWeights(i);
            std::cout << "Current Bias\n\n";
            model.printBias(i);
        }
    }
    

    return 0;
}
