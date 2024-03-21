#include "tensor.cuh"

namespace py = pybind11;

template<typename Dtype>
void bind_tensor(py::module &m, const char *name) {

    py::enum_<BackendType>(m, "BackendType")
        .value("CPU", BackendType::CPU)
        .value("CUDA", BackendType::CUDA)
        .export_values();

    // Bind Tensor class
    py::class_<Tensor<Dtype>>(m, "Tensor", py::buffer_protocol())
        .def(py::init<py::array_t<Dtype>&, BackendType>(),
            py::arg("np_array"),
            py::arg("backend"))

        .def("to_numpy", &Tensor<Dtype>::to_numpy)
        .def("device", &Tensor<Dtype>::device)
        .def("shape", &Tensor<Dtype>::shape)
        .def("__add__", &Tensor<Dtype>::operator+, py::is_operator())
        ;
}

PYBIND11_MODULE(tensor, m) {
    bind_tensor<float>(m, "Tensor");
}


