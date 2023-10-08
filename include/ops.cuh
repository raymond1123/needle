#ifndef __OPS_CUH__
#define __OPS_CUH__

#include "common.hpp"
namespace needle {

template<typename Dtype>
void vec_add_wrapper(Dtype *a, Dtype *b, Dtype *c, size_t n);

template<typename Dtype>
class EwiseAdd {
public:
    virtual void compute_cpu(Dtype *a, Dtype *b, Dtype *c, size_t n) {
    for (size_t i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
    }

    virtual void compute_gpu(Dtype *a, Dtype *b, Dtype *c, size_t n) {
        vec_add_wrapper(a, b, c, n);
        cudaDeviceSynchronize(); // Wait for the GPU kernel to finish
    }
};

}

#endif

