#include "tensor.hpp"

namespace py = pybind11;

// Define bindings for the Tensor class
template<typename Dtype>
void bind_tensor(py::module &m, const char *name) {

    py::enum_<BackendType>(m, "BackendType")
        .value("CPU", BackendType::CPU)
        .value("CUDA", BackendType::CUDA)
        .export_values();

    py::class_<Tensor<Dtype>>(m, name)
        .def(py::init<py::array_t<Dtype>&, BackendType>())
        .def("to_numpy", &Tensor<Dtype>::to_numpy)
        .def("shape", &Tensor<Dtype>::shape)
        ;
}

// Create the Pybind11 module
PYBIND11_MODULE(tensor, m) {
    bind_tensor<float>(m, "Tensor");
}


