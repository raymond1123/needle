#ifndef __INIT_BASIC_CUH__
#define __INIT_BASIC_CUH__

#include "tensor.cuh"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>

namespace py = pybind11;

// Generate uniformly distributed random numbers
template<typename Dtype>
py::array_t<Dtype> _generate_uniform(std::vector<int32_t>& shape, 
                                    Dtype min=static_cast<Dtype>(0.0), 
                                    Dtype max=static_cast<Dtype>(1.0)) {
    size_t size=1;
    for(auto& s: shape)
        size *= s;

    // Initialize a random number generator
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<Dtype> dist(min, max);

    // Create a buffer to hold the random numbers
    auto result = py::array_t<Dtype>(size);

    auto ptr = result.mutable_data();
    for (int i = 0; i < size; ++i) {
        ptr[i] = dist(rng);
    }

    return result;
}

template<typename Dtype>
py::array_t<Dtype> _generate_randb(std::vector<int32_t>& shape, 
                                   float prob=0.5,
                                   Dtype min=static_cast<Dtype>(0.0), 
                                   Dtype max=static_cast<Dtype>(1.0)) {
    size_t size=1;
    for(auto& s: shape)
        size *= s;

    // Initialize a random number generator
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<Dtype> dist(min, max);

    // Create a buffer to hold the random numbers
    auto result = py::array_t<Dtype>(size);

    auto ptr = result.mutable_data();
    for (int i = 0; i < size; ++i) {
        if(dist(rng)>=prob) ptr[i] = static_cast<Dtype>(1.0);
        else ptr[i] = static_cast<Dtype>(0.0);
    }

    return result;
}


// Generate uniformly distributed random numbers
template<typename Dtype>
Tensor<Dtype> rand(std::vector<int32_t> shape, 
                   Dtype min=static_cast<Dtype>(0.0), 
                   Dtype max=static_cast<Dtype>(1.0),
                   BackendType device=BackendType::CUDA) {

    auto result = _generate_uniform<Dtype>(shape, max, min);
    auto tensor = Tensor<Dtype>(result, device);

    return tensor.reshape(shape);
}

// Generate Gaussian distributed random numbers
template<typename Dtype>
Tensor<Dtype> randn(std::vector<int32_t> shape, 
                   Dtype mean=static_cast<Dtype>(0.0), 
                   Dtype std=static_cast<Dtype>(1.0),
                   BackendType device=BackendType::CUDA) {

    size_t size=1;
    for(auto& s: shape)
        size *= s;

    // Initialize a random number generator
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<Dtype> dist(mean, std);

    // Create a buffer to hold the random numbers
    auto result = py::array_t<Dtype>(size);

    auto ptr = result.mutable_data();
    for (int i = 0; i < size; ++i) {
        ptr[i] = dist(rng);
    }

    auto tensor = Tensor<Dtype>(result, device);
    return tensor.reshape(shape);
}

template<typename Dtype>
Tensor<Dtype> randb(std::vector<int32_t> shape, 
                    float prob=0.5,
                    BackendType device=BackendType::CUDA) {

    auto result = _generate_randb<Dtype>(shape, prob);
    auto tensor = Tensor<Dtype>(result, device);
    return tensor.reshape(shape);
}

template<typename Dtype>
Tensor<Dtype> ones(std::vector<int32_t> shape, BackendType device) {
    return Tensor<Dtype>::ones(shape, device);
}

template<typename Dtype>
Tensor<Dtype> ones_like(Tensor<Dtype>& input) {
    return Tensor<Dtype>::ones(input.shape(), input.device());
}

template<typename Dtype>
Tensor<Dtype> zeros(std::vector<int32_t> shape, BackendType device) {
    return Tensor<Dtype>::zeros(shape, device);
}

template<typename Dtype>
Tensor<Dtype> zeros_like(Tensor<Dtype>& input) {
    return Tensor<Dtype>::zeros(input.shape(), input.device());
}

template<typename Dtype>
Tensor<Dtype> constant(std::vector<int32_t> shape, 
                       Dtype val,
                       BackendType device=BackendType::CUDA) {
    return Tensor<Dtype>::fill_val(shape, val, device);
}

template<typename Dtype>
Tensor<Dtype> one_hot(int32_t size, int idx,
                      BackendType device=BackendType::CUDA) {

    idx = idx>=0?idx:size+idx;

    auto p_arr = py::array_t<Dtype>(size);
    auto ptr = p_arr.mutable_data();

    for (int i = 0; i < size; ++i)
        ptr[idx] = static_cast<Dtype>(0.0);

    ptr[idx] = static_cast<Dtype>(1.0);
    std::vector<int32_t> shape = {1,size};

    auto tensor = Tensor<Dtype>(p_arr, device);
    return tensor.reshape(shape);
}

#endif

