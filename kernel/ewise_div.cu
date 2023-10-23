#include "ewise_div.cuh"

namespace needle {

/* EwiseAdd */
template<typename Dtype>
static __global__ void vec_div(Dtype *a, Dtype *b, Dtype *c, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] / b[idx];
}

template<typename Dtype>
void vec_div_wrapper(Dtype *a, Dtype *b, Dtype *c, size_t n) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    vec_div<<<grid_size, block_size>>>(a, b, c, n);
}

template __global__ void vec_div<float>(float *a, float *b, 
                                        float *c, size_t n);

template void vec_div_wrapper<float>(float *a, float *b, 
                                     float *c, size_t n);

} //namespace needle

