#ifndef __INITIAL_CUH__
#define __INITIAL_CUH__

#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>

#include "tensor.cuh"
#include "init/init_basic.cuh"

namespace py = pybind11;

template<typename Dtype>
Tensor<Dtype> xavier_uniform(std::vector<int32_t> shape, 
                             Dtype gain=1.0,
                             BackendType device=BackendType::CUDA) {

    assert(shape.size()==2 && "dimension of xavier_uniform shape must be 2");

    auto scope = [](Dtype fan_in, Dtype fan_out) { 
        float root = 6.0/(float(fan_in)+float(fan_out));
        return sqrt(root);
    };

    Dtype high = gain*static_cast<Dtype>(scope(shape[0], shape[1]));

    return rand(shape, -high, high, device);
}

template<typename Dtype>
Tensor<Dtype> xavier_normal(std::vector<int32_t> shape, 
                             Dtype gain=1.0,
                             BackendType device=BackendType::CUDA) {

    assert(shape.size()==2 && "dimension of xavier_uniform shape must be 2");

    auto std_f = [](Dtype fan_in, Dtype fan_out) { 
        float root = 2.0/(float(fan_in)+float(fan_out));
        return sqrt(root);
    };

    Dtype std = gain*static_cast<Dtype>(std_f(shape[0], shape[1]));

    return randn(shape, static_cast<Dtype>(0.0), std, device);
}

template<typename Dtype>
Tensor<Dtype> kaiming_uniform(std::vector<int32_t> shape, 
                              BackendType device=BackendType::CUDA,
                              std::string nonlinearity="relu") {

    assert(shape.size()==2 && "dimension of xavier_uniform shape must be 2");
    assert(nonlinearity=="relu" && "only relu supported currently");

    auto scope = [](Dtype fan_in) { 
        float gain = sqrt(2.0);
        return gain*sqrt(3.0/(float)fan_in);
    };

    Dtype high = static_cast<Dtype>(scope(shape[0]));

    return rand(shape, -high, high, device);
}

template<typename Dtype>
Tensor<Dtype> kaiming_normal(std::vector<int32_t> shape, 
                             BackendType device=BackendType::CUDA,
                             std::string nonlinearity="relu") {

    assert(shape.size()==2 && "dimension of xavier_uniform shape must be 2");
    assert(nonlinearity=="relu" && "only relu supported currently");

    auto std_f = [](Dtype fan_in) { 
        float gain = sqrt(2.0);
        return gain/sqrt((float)fan_in);
    };

    Dtype std = static_cast<Dtype>(std_f(shape[0]));

    return randn(shape, static_cast<Dtype>(0.0), std, device);
}

#endif

