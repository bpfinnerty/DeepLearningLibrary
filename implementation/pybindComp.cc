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
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include "Net.h"

namespace py = pybind11;

PYBIND11_MODULE(projNet,m){  
 py::class_<Net>(m,"Net")
   .def(py::init<>())
   .def("setInput",&Net::setInput)
   .def("ff",&Net::ff)
   .def("addLayer",&Net::addLayer)
   .def("setLearningRate",&Net::setLearningRate)
   .def("relu",&Net::relu)
   .def("backwardStep",&Net::backwardStep)
   .def("leakyRelu",&Net::leakyRelu)
   .def("updateWeights",&Net::updateWeights)
   .def("crossEntropy",&Net::crossEntropy)
   .def("zeroGrad",&Net::zeroGrad)
   .def("softMax",&Net::softMax)
   .def("printWeights",&Net::printWeights)
   .def("Sigmoid",&Net::Sigmoid)
  .def("printBias",&Net::printBias)
   .def("setWeights",&Net::setWeights)
   .def("setBias",&Net::setBias)
   .def("printGrad",&Net::printGrad)
   .def("printBiasGrad",&Net::printBiasGrad)
   .def("msLoss",&Net::msLoss)
   .def("setThreadNum",&Net::setThreadNum)
  .def("printDim",&Net::printDim);
}