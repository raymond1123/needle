#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include "backend/base_tensor.hpp"

namespace py = pybind11;

template<typename Dtype>
class Tensor {
public:
    Tensor(py::array_t<Dtype>& np_array, BackendType backend):
        __backend(backend) {

        // Initialize the backend based on the specified type
        if (__backend == BackendType::CPU) {
            __tensor = std::make_shared<CpuTensor<Dtype>>(np_array);
        } else if (__backend == BackendType::CUDA) {
            __tensor = std::make_shared<CudaTensor<Dtype>>(np_array);
        } else {
            throw std::runtime_error("Unsupported backend type.");
        }
    }

    py::array_t<Dtype> to_numpy() {
        return __tensor->to_numpy();
    }

    std::vector<size_t> shape() {
        return __tensor->shape();
    }

private:
    BackendType __backend;
    std::shared_ptr<BaseTensor<Dtype>> __tensor;
};

#endif

