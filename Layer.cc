#include "Layer.h"


// Methods for Layer Class
void Layer::normal_distribution_weights(){
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


std::vector<double> Layer::ff(std::vector<double> x){
    
    if ((int)x.size() != inputs){
        throw std::runtime_error("Mismatched dims in index");
    }
    else{
        std::cout << "Input is length " << x.size() << "\n";
    }
    std::vector<double> ret(outputs,0.0);
    
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
    #pragma omp parallel for num_threads(Net::numThreads)
    for(int i = 0; i < inputs; ++i){
        std::cout << "Starting Node " << i << "\n";
        for(int j = 0; j<outputs; ++j){
            #pragma omp critical
            {
                r[i] += x[i]*weights[i*outputs+j];
            }
            //std::cout << "Completed weight " << j << "\n";  
        }
        
        nodeOutput[i] = r[i];
    }
    std::cout << "finished multiply\n";
    return ret;
    
}