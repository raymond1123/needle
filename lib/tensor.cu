#include "tensor.cuh"
#include "needle_util.cuh"
#include "nn/function.cuh"
#include "nn/nn_basic.cuh"
#include "init/init_basic.cuh"
#include "init/initial.cuh"

namespace py = pybind11;

template<typename Dtype>
void bind_operator_iplus_tensor(py::class_<Tensor<Dtype>>& tensor_class) {
    tensor_class.def("__iadd__",
        [](Tensor<Dtype>& self, Tensor<Dtype>& other) {
            self += other;
            return self;
        });
}

template<typename Dtype>
void bind_operator_plus_tensor(py::class_<Tensor<Dtype>>& tensor_class) {
    tensor_class.def("__add__",
        [](Tensor<Dtype>& self, Tensor<Dtype>& other) {
            return self + other;
        });
}

// Binding for operator+ with a scalar
template<typename Dtype>
void bind_operator_plus_scalar(py::class_<Tensor<Dtype>>& tensor_class) {
    tensor_class.def("__add__",
        [](Tensor<Dtype>& self, const Dtype scalar) {
            return self + scalar;
        });
}

template<typename Dtype>
void bind_operator_sub_tensor(py::class_<Tensor<Dtype>>& tensor_class) {
    tensor_class.def("__sub__",
        [](Tensor<Dtype>& self, Tensor<Dtype>& other) {
            return self - other;
        });
}

// Binding for operator+ with a scalar
template<typename Dtype>
void bind_operator_sub_scalar(py::class_<Tensor<Dtype>>& tensor_class) {
    tensor_class.def("__sub__",
        [](Tensor<Dtype>& self, const Dtype scalar) {
            return self - scalar;
        });
}

template<typename Dtype>
void bind_operator_mul_tensor(py::class_<Tensor<Dtype>>& tensor_class) {
    tensor_class.def("__mul__",
        [](Tensor<Dtype>& self, Tensor<Dtype>& other) {
            return self * other;
        });
}

// Binding for operator+ with a scalar
template<typename Dtype>
void bind_operator_mul_scalar(py::class_<Tensor<Dtype>>& tensor_class) {
    tensor_class.def("__mul__",
        [](Tensor<Dtype>& self, const Dtype scalar) {
            return self * scalar;
        });
}

template<typename Dtype>
void bind_operator_div_tensor(py::class_<Tensor<Dtype>>& tensor_class) {
    tensor_class.def("__truediv__",
        [](Tensor<Dtype>& self, Tensor<Dtype>& other) {
            return self / other;
        });
}

// Binding for operator+ with a scalar
template<typename Dtype>
void bind_operator_div_scalar(py::class_<Tensor<Dtype>>& tensor_class) {
    tensor_class.def("__truediv__",
        [](Tensor<Dtype>& self, const Dtype scalar) {
            return self / scalar;
        });
}

template<typename Dtype>
void bind_operator_pow_tensor(py::class_<Tensor<Dtype>>& tensor_class) {
    tensor_class.def("__pow__",
        [](Tensor<Dtype>& self, Tensor<Dtype>& other) {
            return self.op_pow(other);
        });
}

template<typename Dtype>
void bind_operator_pow_scalar(py::class_<Tensor<Dtype>>& tensor_class) {
    tensor_class.def("__pow__",
        [](Tensor<Dtype>& self, const Dtype scalar) {
            return self.op_pow(scalar);
        });
}

template<typename Dtype>
void bind_tensor(py::module &m, const char *name) {

    // Binding for operator+ with another Tensor
    py::enum_<BackendType>(m, "BackendType")
        .value("CPU", BackendType::CPU)
        .value("CUDA", BackendType::CUDA)
        .export_values();

    // Bind Tensor class
    py::class_<Tensor<Dtype>> tensor_class(m, name);
    tensor_class
        .def(py::init<py::array_t<Dtype>&, BackendType>(),
            py::arg("np_array"),
            py::arg("backend"))

        .def("reshape", &Tensor<Dtype>::reshape)
        .def("flip", &Tensor<Dtype>::flip)
        .def("__getitem__", &Tensor<Dtype>::slice)
        .def("__setitem__", &Tensor<Dtype>::setitem)
        .def("broadcast_to", &Tensor<Dtype>::broadcast_to)
        .def("permute", &Tensor<Dtype>::permute)
        .def("transpose", &Tensor<Dtype>::transpose)
        .def("sum", (Tensor<Dtype> (Tensor<Dtype>::*)(std::vector<int>)) &Tensor<Dtype>::summation, "Summation with specified axes")
        .def("sum", (Tensor<Dtype> (Tensor<Dtype>::*)()) &Tensor<Dtype>::summation, "Summation without specified axes")
        .def("__matmul__", &Tensor<Dtype>::matmul)
        .def("dilate", &Tensor<Dtype>::dilate, py::arg("dilation"), py::arg("axes"))
        .def("to_numpy", &Tensor<Dtype>::to_numpy)
        .def("device", &Tensor<Dtype>::device)
        .def("shape", &Tensor<Dtype>::shape)
        .def("size", &Tensor<Dtype>::size)
        .def("strides", &Tensor<Dtype>::strides)
        .def("offset", &Tensor<Dtype>::offset)
        .def("from_buffer", &Tensor<Dtype>::from_buffer)
        .def("contiguous", &Tensor<Dtype>::contiguous)
        .def("backward", &Tensor<Dtype>::backward)
        .def("grad", &Tensor<Dtype>::grad)
        ;

    /* operators */
    bind_operator_plus_tensor(tensor_class);
    bind_operator_plus_scalar(tensor_class);

    bind_operator_sub_tensor(tensor_class);
    bind_operator_sub_scalar(tensor_class);

    bind_operator_mul_tensor(tensor_class);
    bind_operator_mul_scalar(tensor_class);

    bind_operator_div_tensor(tensor_class);
    bind_operator_div_scalar(tensor_class);

    bind_operator_pow_tensor(tensor_class);
    bind_operator_pow_scalar(tensor_class);

    bind_operator_iplus_tensor(tensor_class);
}

template<typename Dtype>
void bind_functional(py::module &m) {
    py::module nn = m.def_submodule("nn", "Neural network operations");
    py::module functional = nn.def_submodule("functional", "Functions used in neural networks");
    functional.def("ccc", &ccc);
    functional.def("ddd", &ddd);
    functional.def("pad", &pad<Dtype>);
}

template<typename Dtype>
void bind_init(py::module &m) {
    m.def("rand", &rand<Dtype>, 
          py::arg("shape"), py::arg("min")=0.0, 
          py::arg("max")=1.0, py::arg("device")=BackendType::CUDA,
          "Generate uniformly distributed random tensor");

    m.def("randn", &randn<Dtype>, 
          py::arg("shape"), py::arg("mean")=0.0, 
          py::arg("std")=1.0, py::arg("device")=BackendType::CUDA, 
          "Generate Gaussian distributed random tensor");

    m.def("randb", &randb<Dtype>, 
          py::arg("shape"), py::arg("p")=0.5, 
          py::arg("device")=BackendType::CUDA, 
          "Generate binary random tensor");

    m.def("ones", &ones<Dtype>, py::arg("shape"), py::arg("device")=BackendType::CUDA);
    m.def("zeros", &zeros<Dtype>, py::arg("shape"), py::arg("device")=BackendType::CUDA);
    m.def("ones_like", &ones_like<Dtype>, py::arg("input"));
    m.def("zeros_like", &zeros_like<Dtype>, py::arg("input"));

    m.def("constant", &constant<Dtype>, py::arg("shape"), py::arg("val"),
                                        py::arg("device")=BackendType::CUDA);
    m.def("one_hot", &one_hot<Dtype>, py::arg("size"), py::arg("idx"),
                                    py::arg("device")=BackendType::CUDA);

    m.def("xavier_uniform", &xavier_uniform<Dtype>, 
          py::arg("shape"), py::arg("gain")=1.0,
          py::arg("device")=BackendType::CUDA);

    m.def("xavier_normal", &xavier_normal<Dtype>, 
          py::arg("shape"), py::arg("gain")=1.0,
          py::arg("device")=BackendType::CUDA);

    m.def("kaiming_uniform", &kaiming_uniform<Dtype>, 
          py::arg("shape"), 
          py::arg("device")=BackendType::CUDA,
          py::arg("nonlinearity")="relu");

    m.def("kaiming_normal", &kaiming_normal<Dtype>, 
          py::arg("shape"), 
          py::arg("device")=BackendType::CUDA,
          py::arg("nonlinearity")="relu");
}

template<typename Dtype>
void bind_module(py::module &m) {
    py::module nn = m.def_submodule("nn", "Neural network operations");

    //std::vector<std::shared_ptr<Module<Dtype>>> Module<Dtype>::_modules;
    py::class_<Module<Dtype>, std::shared_ptr<Module<Dtype>>>(nn, "Module")
       .def(py::init<>())
       .def(py::init<std::vector<std::shared_ptr<Module<Dtype>>> &>())
       .def("__call__", &Module<Dtype>::operator())
       .def("train", &Module<Dtype>::train)
       .def("eval", &Module<Dtype>::eval)
       .def("forward", &Module<Dtype>::forward);

    py::class_<Sequential<Dtype>, Module<Dtype>, 
                std::shared_ptr<Sequential<Dtype>>>(nn, "Sequential")
        .def(py::init<std::vector<std::shared_ptr<Module<Dtype>>> &>())
        .def("forward", &Sequential<Dtype>::forward);

    py::class_<Linear<Dtype>, Module<Dtype>, 
                std::shared_ptr<Linear<Dtype>>>(nn, "Linear")
        .def(py::init<int, int>())
        .def("forward", &Linear<Dtype>::forward);

}


PYBIND11_MODULE(tensor, m) {
    /* tensor */
    bind_tensor<float>(m, "Tensor");

    m.def("stack", &stack<float>, py::arg("inputs"), py::arg("dim")=0);
    m.def("split", &split<float>, py::arg("input"), py::arg("dim")=0);

    /* nn.functional */
    bind_functional<float>(m);

    /* nn.init */
    bind_init<float>(m);

    /* nn.module */
    bind_module<float>(m);
}

/*
PYBIND11_MODULE(tensor, m) {
    bind_tensor<double>(m, "Tensor");

    m.def("ones", &ones<double>, py::arg("shape"), py::arg("backend"));
    m.def("zeros", &zeros<double>, py::arg("shape"), py::arg("backend"));
}
*/

