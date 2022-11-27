#include "Net.h"
#include <vector>
#include <cstdlib>

int main(){
    Net model = Net();

    model.addLayer(3,6);
    model.addLayer(6,4);
    model.addLayer(4,2);

    model.printDim(0);
    model.printDim(1);
    model.printDim(2);

    model.writeWeightsToFile();

    std::cout << "printing weights that were written to file\n";
    model.printWeights(0);
    model.printWeights(1);
    model.printWeights(2);

    std::cout << "printing bias that were written to file\n";
    model.printBias(0);
    model.printBias(1);
    model.printBias(2);

    std::cout << "reading weights and bias from file net_weights_bias_mod.txt\n";

    model.readWeightsFromFile("net_weights_bias_mod.txt");

    std::cout << "printing weights that were read from file\n";
    model.printWeights(0);
    model.printWeights(1);
    model.printWeights(2);

    std::cout << "printing bias that were read from file\n";
    model.printBias(0);
    model.printBias(1);
    model.printBias(2);



    return 0;
}
