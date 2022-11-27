# DeepLearningLibrary

DeepLearningLibrary is a simple C++ based Deep Learning library that can be used in order to create and train neural nets of varying sizes and used for tasks such as classification of data.  Included in this library is a built-in CSV data parser that extracts and formats the data for use with this library.  DeepLearningLibrary can also be utilized not just in a native C++ application, but can be used in Python projects as well.  We used the pybind11 module in order to create a wrapper class for the C++ functions.  

##Pre-requisites
In order to fully utilize the DeepLearningLibrary files, the ```C++ Boost``` library is needed in order for some of the functionality to work correctly.  ```Boost``` is mainly a header only library, so no building of the library is needed.  In order to install Boost, follow the steps as outlined here: https://www.boost.org/doc/libs/1_80_0/more/getting_started/unix-variants.html.  We have included an older version of the ```Boost``` directory that was originally used here, however if that does not work, following these installation steps will allow you to retrieve the updated version. 

In order to utilize DeepLearningLibrary in a Python project, ```pybind11``` must be installed onto your system first.  There are various ways in which ```pybind11``` can be installed, follow the instructions here: https://pybind11.readthedocs.io/en/latest/installing.html.

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

