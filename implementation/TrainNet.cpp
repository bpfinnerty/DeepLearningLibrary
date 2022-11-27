#include "Net.h"
#include "data_input.h"
#include <vector>
#include <cstdlib>


std::vector<double> forward(std::vector<double> val, Net neuralNet){
    std::vector<double> ret = neuralNet.ff(val, 0);
    ret = neuralNet.Sigmoid(ret);
    ret = neuralNet.ff(ret, 1);
    ret = neuralNet.Sigmoid(ret);
    ret = neuralNet.ff(ret, 2);
    // ret = neuralNet.Sigmoid(ret);
    // ret = neuralNet.ff(ret, 3);
    // ret = neuralNet.Sigmoid(ret);
    // ret = neuralNet.ff(ret, 4);
    ret = neuralNet.softMax(ret);
    return ret;
}

std::vector<std::vector<double>> oneHotEncode(std::vector<int> labels, int num_of_classes){
    std::vector<std::vector<double>> zeroed(labels.size(), std::vector<double>(num_of_classes, 0));
    for(int i = 0; i < labels.size(); i++){
        int label_val = labels.at(i);
        zeroed[i][label_val] = 1;
    }
    return zeroed;
}

int main(){
    std::cout << "start of Main\n";
    float learningRate = 0.02;
    int numThreads = 1;
    int epoch = 300;
    int batch = 8;

    std::string trainingDataPath = "mnist_train.csv";
    std::string testingDataPath = "winequality-red.csv";

    std::cout << "getting to this point.\n";

    data_input dataReader = data_input(trainingDataPath, ",");
    dataframe data = dataReader.getData(0);
    std::cout << "opened data\n";

    Net model = Net();
    model.addLayer(784, 128);
    model.addLayer(128, 64);
    model.addLayer(64, 10);

    model.setLearningRate(learningRate);

    // model.printDim(0);
    // model.printDim(1);
    // model.printDim(2);
    // model.printDim(3);
    // model.printDim(4);
    std::cout << "Set initial values\n";

    for(int i = 0; i < epoch; i++){
        std::cout << "epoch: " << i << "\n";

        dataframe shuffled = data.getSample(0.1);
        std::vector<int> truth = shuffled.getLabels();
        std::vector<std::vector<double>> train = shuffled.getData();
        std::vector<std::vector<double>> encoded_truth = oneHotEncode(truth, 10);

        int max_examples = encoded_truth.size();
        int index = 0;
        double avgLoss = 0;

        while(index < max_examples){
            model.zeroGrad();
            int counter = 0;

            while(counter < batch){
                if(index >= max_examples){
                    break;
                }
                std::cout << "in the while loop";

                std::vector<double> predictions = forward(train[index], model);
                double loss = model.crossEntropy(predictions, encoded_truth[index]);
                avgLoss += loss;
                model.backwardStep(predictions, encoded_truth[index]);
                index += 1;
                counter += 1;

                // model.printGrad();
            }

            model.updateWeights();
        }
        std::cout << "Avg Loss: " << avgLoss/max_examples << " for epoch: " << i << "\n";
    }

    return 0;
}
