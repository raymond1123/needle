#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For handling std::vector
#include <pybind11/stl_bind.h> // For binding std::string
#include <pybind11/numpy.h>

#include <vector>
#include <iostream>
#include <string>

#include "tensor.hpp"

namespace py = pybind11;

template <typename Dtype>
void declareTensor(py::module &m, const std::string& class_name) {
    py::class_<needle::NDArray::Tensor<Dtype>>(m, class_name.c_str())
        .def(py::init<const py::list&, const uint32_t, const std::string>(),
             py::arg("data"), py::arg("offset"), py::arg("device")="cuda")
        .def("shape", &needle::NDArray::Tensor<Dtype>::shape)
        .def("strides", &needle::NDArray::Tensor<Dtype>::strides)
        .def("device", &needle::NDArray::Tensor<Dtype>::device)
        .def("ndim", &needle::NDArray::Tensor<Dtype>::ndim)
        .def("size", &needle::NDArray::Tensor<Dtype>::size)
        .def("offset", &needle::NDArray::Tensor<Dtype>::offset);
}

PYBIND11_MODULE(unittest, m) {
    declareTensor<int>(m, "TensorInt");
    declareTensor<float>(m, "TensorFloat32");
    // Add more instantiations for different Dtype types as needed
}


