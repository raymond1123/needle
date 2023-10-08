#include "ops.cuh"

namespace needle {

/* EwiseAdd */
template<typename Dtype>
static __global__ void vec_add(Dtype *a, Dtype *b, Dtype *c, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

template<typename Dtype>
void vec_add_wrapper(Dtype *a, Dtype *b, Dtype *c, size_t n) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    //vec_add<Dtype><<<grid_size, block_size>>>(a, b, c, n);
    vec_add<<<grid_size, block_size>>>(a, b, c, n);
}

template __global__ void vec_add<float>(float *a, float *b, 
                                        float *c, size_t n);
template void vec_add_wrapper<float>(float *a, float *b, 
                                     float *c, size_t n);

/* EwiseMinus */
template<typename Dtype>
static __global__ void vec_minus(Dtype *a, Dtype *b, Dtype *c, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

template<typename Dtype>
void vec_minus_wrapper(Dtype *a, Dtype *b, Dtype *c, size_t n) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    //vec_add<Dtype><<<grid_size, block_size>>>(a, b, c, n);
    vec_add<<<grid_size, block_size>>>(a, b, c, n);
}

template __global__ void vec_minus<float>(float *a, float *b, 
                                          float *c, size_t n);
template void vec_minus_wrapper<float>(float *a, float *b, 
                                       float *c, size_t n);

} //namespace needle

