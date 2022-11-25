#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include <cmath>
#include<vector>
#include<math.h>
#include<stdexcept>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include "Layer.h"
#include "Net.h"


namespace py = pybind11;

PYBIND11_MODULE(projNet,m){
  py::class_<Layer>(m,"Layer")
    .def(py::init<int,int>())
    .def("ff",&Layer::ff);
  
  py::class_<Net>(m,"Net")
    .def(py::init<>())
    .def("setLearningRate",&Net::setLearningRate)
    .def("setInputs",&Net::setInputs)
    .def("relu",&Net::relu)
    .def("backwardStep",&Net::backwardStep)
    .def("leakyRelu",&Net::leakyRelu)
    .def("updateWeights",&Net::updateWeights)
    .def("crossEntropy",&Net::crossEntropy)
    .def("setThreads",&Net::setThreads)
    .def("zeroGrad",&Net::zeroGrad);
  
}