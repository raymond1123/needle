#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For handling std::vector
#include <pybind11/stl_bind.h> // For binding std::string
#include <pybind11/numpy.h>

#include <vector>
#include <tuple>
#include <iostream>
#include <string>

#include "tensor.hpp"

namespace py = pybind11;

template<typename Dtype>
needle::Tensor<Dtype> ones(const py::tuple &shape,
                           const std::string device="cuda") {

    std::vector<size_t> shape_vec;
    for(size_t i=0; i<shape.size(); ++i)
        shape_vec.push_back(shape[i].cast<size_t>());

    needle::Tensor<Dtype> res = needle::Tensor<Dtype>(shape_vec, device);
    return res.all_ones();
}

template <typename Dtype>
void declareTensor(py::module &m, const std::string& class_name) {
    py::class_<needle::Tensor<Dtype>>(m, class_name.c_str())
        .def(py::init<const py::list&, const std::string>(),
             py::arg("data"), py::arg("device")="cuda")
        .def(py::init<const std::vector<size_t>&, const std::string>(),
             py::arg("shape"), py::arg("device")="cuda")
        .def("numpy", &needle::Tensor<Dtype>::numpy)
        .def("__add__", &needle::Tensor<Dtype>::operator+)
        .def("__sub__", &needle::Tensor<Dtype>::operator-)
        .def("__mul__", &needle::Tensor<Dtype>::operator*)
        .def("__truediv__", &needle::Tensor<Dtype>::operator/)
        .def("__repr__", &needle::Tensor<Dtype>::print);
}

PYBIND11_MODULE(Unittest, m) {
    declareTensor<float>(m, "Tensor");
    m.def("ones", &ones<float>, "Create a tensor of ones",
          py::arg("shape"), py::arg("device") = "cuda");
}

