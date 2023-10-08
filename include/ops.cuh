#ifndef __OPS_CUH__
#define __OPS_CUH__

#include "common.hpp"
namespace needle {

template<typename Dtype>
class TensorOP {
public:
    virtual void compute_cpu(Dtype *a, Dtype *b, Dtype *c, size_t n)=0;
    virtual void compute_gpu(Dtype *a, Dtype *b, Dtype *c, size_t n)=0;
};

template<typename Dtype>
void vec_add_wrapper(Dtype *a, Dtype *b, Dtype *c, size_t n);

template<typename Dtype>
class EwiseAdd: TensorOP<Dtype> {
public:
    virtual void compute_cpu(Dtype *a, Dtype *b, Dtype *c, size_t n) override {
    for (size_t i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
    }

    virtual void compute_gpu(Dtype *a, Dtype *b, Dtype *c, size_t n) override {
        vec_add_wrapper(a, b, c, n);
        cudaDeviceSynchronize(); // Wait for the GPU kernel to finish
    }
};

template<typename Dtype>
class EwiseMinus: TensorOP<Dtype> {
public:
    virtual void compute_cpu(Dtype *a, Dtype *b, Dtype *c, size_t n) override {
    for (size_t i = 0; i < n; ++i)
        c[i] = a[i] - b[i];
    }

    virtual void compute_gpu(Dtype *a, Dtype *b, Dtype *c, size_t n) override {
        vec_minus_wrapper(a, b, c, n);
        cudaDeviceSynchronize(); // Wait for the GPU kernel to finish
    }
};

}

#endif

