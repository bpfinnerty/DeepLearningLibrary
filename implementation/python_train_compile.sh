g++ -O3 -shared -std=c++11  -Wall -fopenmp -fPIC  $(python3 -m pybind11 --includes) Net.cc pybindComp.cc -o projNet$(python3-config --extension-suffix)
