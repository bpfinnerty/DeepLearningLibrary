# DeepLearningLibrary

DeepLearningLibrary is a simple C++ based Deep Learning library that can be used in order to create and train neural nets of varying sizes and used for tasks such as classification of data.  Included in this library is a built-in CSV data parser that extracts and formats the data for use with this library.  DeepLearningLibrary can also be utilized not just in a native C++ application, but can be used in Python projects as well.  We used the pybind11 module in order to create a wrapper class for the C++ functions.  

##Pre-requisites
In order to fully utilize the DeepLearningLibrary files, the ```C++ Boost``` library is needed in order for some of the functionality to work correctly.  ```Boost``` is mainly a header only library, so no building of the library is needed.  In order to install Boost, follow the steps as outlined here: https://www.boost.org/doc/libs/1_80_0/more/getting_started/unix-variants.html.  We have included an older version of the ```Boost``` directory that was originally used here, however if that does not work, following these installation steps will allow you to retrieve the updated version. 

In order to utilize DeepLearningLibrary in a Python project, ```pybind11``` must be installed onto your system first.  There are various ways in which ```pybind11``` can be installed, follow the instructions here: https://pybind11.readthedocs.io/en/latest/installing.html. You also need to install `conda-forge python-devtools`. 

If you wish to run our PyTorch model as a baseline, then you must also install pytorch (https://pytorch.org/) as well as Pandas and Numpy.

##Installation
To use the library as a whole, including the ```Boost``` and ```pybind11``` libraries, simply copy the following directories: ```boost_1_62_0```, ```data``` and ```implementation```.  These should contain all you need in order to utilize the library.  
You can also simply clone the repository as follows:
```
https://github.com/bpfinnerty/DeepLearningLibrary.git
```
Once that is done, you can do one of the following.
To use in a C++ project, simply include the necessary header files into your source files in this fashion:
```cpp
#include "implementation/data_input.h" // If you want to use the built in CSV parser instead
#include "implementation/Net.h" // Contains the actual functions used for creating the Neural Nets

int main(){
...
}
```
Once that is done, you can compile as follows:
```
g++ -g -O3 -fopenmp -g -std=c++11 -Wall  exampleUsage.cpp data_input.cpp Net.cc -o exampleExecutable
```

In order to use this in a Python project, the ```pybindComp.cc``` file is needed in order to properly create the python module that can be imported to a Python project.  This can be done as follows:
```
g++ -O3 -shared -std=c++11  -Wall -fopenmp -fPIC  $(python3 -m pybind11 --includes) Net.cc pybindComp.cc -o examplePythonModule$(python3-config --extension-suffix)
```
Once this is done, a Python module is created that can then be imported into an existing Python project.

##Usage

Once the appropriate headers have been added to your C++ soure file, you can instantiate an object of the ```Net``` class.
```cpp
Net model = Net();
```

From there, you can add however many layers you would like to use, specifying the number of inputs and outputs for each layer.
```cpp
model.addLayer(784, 128);
model.addLayer(128, 64);
model.addLayer(64, 10);
```

You can set the learning rate that the net will use.
```cpp
model.setLearningRate(0.02);
```

We use OpenMP to accelerate some of the computations being done by doing them in parallel. You can set the number of threads being used as follos.

```cpp
model.setThreadNum(num_of_threads); 
```

Once this is done, a variety of functions can be called on the layers of the model, they are as follows:
```cpp
model.ff(std::vector<double> vals, int layer_number);
model.relu(std::vector<double> vals);
model.leakyRelu(std::vector<double> vals);
model.Sigmoid(std::vector<double> vals);
model.softMax(std::vector<double> vals);
```


Once you have trained a model you can save it's weights and biases to file and later read that file to load your model:
```cpp
model.writeWeightsToFile();
model.readWeightsFromFile(std::string filename);
```

On top of that we have a few example scripts you can play around with. First is our Pytorch implementation `PyTorchExample.py`. 
You can also use `./train_net_compile.sh` to run our c++ implementation. In addition there is `./python_train_compile.sh` to run
our Python implementation. You can modify PyTorchExample.py and PyTorchDefinition.py to change the model and training there. 
The same can be done for TrainNet.CPP C++ and `TrainNet.py and NetDefinition.py`.

In all of these files you can adjust the number of layers, learning rate, threads by modifying the the files themselves.

To Run (After compiling):

- Pytorch: `python3 PyTorchExample.py`
- C++: `./TrainNetCPP`
- Python: `python3 TrainNet.py`



## Functions
```cpp
Net::setThreadNum(int num);
Net::setInput(std::vector<double> x);
Net::addLayer(int inputs, int outputs);
Net::setWeights(std::vector<double> startWeights, int layer);
Net::setBias(std::vector<double> startBias, int layer);
Net::printDim(int layer);
Net::relu(std::vector<double> x);
Net::leakyRelu(std::vector<double> x)
Net::msLoss(std::vector<double> x,std::vector<double> target)
Net::crossEntropy(std::vector<double> output,std::vector<double> target)
Net::softMax(std::vector<double> x);
Net::zeroGrad();
Net::backwardStep(std::vector<double> output,std::vector<double> target);
Net::printWeights(int layer);
Net::writeWeightsToFile();
Net::readWeightsFromFile(std::string filename);
Net::printBias(int layer);
Net::printGrad();
Net::printBiasGrad();
Net::updateWeights();
Net::Sigmoid(std::vector<double> x);
Net::ff(std::vector<double> x, int layer);
data_input::getData(int col_index_of_label);
data_input::getLabels();
```

