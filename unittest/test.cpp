#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For handling std::vector
#include <pybind11/stl_bind.h> // For binding std::string
#include <pybind11/numpy.h>

#include <vector>
#include <iostream>
#include <string>

#include "ndarray.hpp"

namespace py = pybind11;

template <typename Dtype>
void declareNDArray(py::module &m, const std::string& class_name) {
    py::class_<needle::NDArray<Dtype>>(m, class_name.c_str())
        .def(py::init<const py::list&, const uint32_t, const std::string>(),
             py::arg("data"), py::arg("offset"), py::arg("device")="cuda")
        .def(py::init<const std::shared_ptr<Memory<Dtype>>, 
             std::vector<size_t>, std::vector<size_t>, 
             const uint32_t, const std::string>(),
             py::arg("data"), py::arg("shape"), py::arg("strides"), 
             py::arg("offset"), py::arg("device")="cuda")
        .def("shape", &needle::NDArray<Dtype>::shape)
        .def("strides", &needle::NDArray<Dtype>::strides)
        .def("device", &needle::NDArray<Dtype>::device)
        .def("ndim", &needle::NDArray<Dtype>::ndim)
        .def("size", &needle::NDArray<Dtype>::size)
        .def("offset", &needle::NDArray<Dtype>::offset)
        .def("__add__", &needle::NDArray<Dtype>::operator+)
        .def("__repr__", &needle::NDArray<Dtype>::print);
}

PYBIND11_MODULE(unittest, m) {
    //declareTensor<int>(m, "TensorInt");
    declareNDArray<float>(m, "TensorFloat32");
    // Add more instantiations for different Dtype types as needed
}


