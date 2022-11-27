g++ -g -O3 -fopenmp -g -std=c++11 -Wall $(python3 -m pybind11 --includes) TrainNet.cpp data_input.cpp Net.cc -o TrainNetCPP
