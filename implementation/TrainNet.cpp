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
    std::cout << "start of Main\n";
    float learningRate = 0.002;
    int numThreads = 8;
    int epoch = 50;
    int batch = 200;

    std::string trainingDataPath = "../data/mnist_train.csv";
    std::string testingDataPath = "../data/mnist_test.csv";

    std::cout << "getting to this point.\n";

    data_input dataReader = data_input(trainingDataPath, ",");
    dataframe data = dataReader.getData(0);

    // test_data_input test_dataReader = data_input(testingDataPath, ",");
    // test_dataframe test_data = test_dataReader.getData(0);
    std::cout << "opened data\n";

    Net model = Net();
    model.addLayer(784, 128);
    model.addLayer(128, 64);
    model.addLayer(64, 10);

    //model.printWeights(1);
    //std::cout << "created Net\n";

    model.setLearningRate(learningRate);
    model.setThreadNum(numThreads);

    // model.printDim(0);
    // model.printDim(1);
    // model.printDim(2);
    // model.printDim(3);
    // model.printDim(4);
    std::cout << "Set initial values\n";

    for(int i = 0; i < epoch; i++){
        std::cout << "epoch: " << i << "\n";

        // dataframe shuffled = data.getSample(1.0);
        std::vector<int> truth = data.getLabels();
        std::vector<std::vector<double>> train = data.getData();
        std::vector<std::vector<double>> encoded_truth = oneHotEncode(truth, 10);
        std::cout << "dataRetrieved\n";
        int max_examples = train.size();
        int index = 0;
        double avgLoss = 0;

        while(index < max_examples){
            // std::cout << "Index: " << index << "\n";
            model.zeroGrad();
            int counter = 0;

            while(counter < batch){
                if(index >= max_examples){
                    break;
                }
                std::vector<double> normalized(784,0.0);
                for(int v = 0; v < 784;++v){
                    normalized[v] = (train[index])[v]/255.0;
                }
                model.setInput(normalized);
                std::vector<double> predictions = forward(normalized, &model);
                // std::cout << "[";
                // for (auto i: predictions)
                //     std::cout << i << ' ';
                // std::cout << "]\n";
                double loss = model.crossEntropy(predictions, encoded_truth[index]);
                avgLoss += loss;
                //std::cout << "Loss: " << loss << " for epoch: " << i << "\n";
                model.backwardStep(predictions, encoded_truth[index]);
                index += 1;
                counter += 1;

                
            }
            
            model.updateWeights();
        }
        std::cout << "Avg Loss: " << avgLoss/max_examples << " for epoch: " << i << "\n";
    }

    return 0;
}
