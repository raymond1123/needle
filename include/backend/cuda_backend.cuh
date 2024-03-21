#ifndef __CUDA_BACKEND_HPP__
#define __CUDA_BACKEND_HPP__

#include "common.hpp"

template<typename Dtype>
class CudaArray {
public:
    CudaArray(const size_t size);
    ~CudaArray() {
        cudaFree(__ptr);
        __size = 0;
    }

    inline size_t size() {return __size;}
    inline Dtype* get_ptr() {return __ptr;}
    cudaError_t device2host(Dtype* host_ptr);
    cudaError_t host2device(const Dtype* host_ptr);

    void set_all_zeros();

private:
    Dtype *__ptr;
    size_t __size;
};

template<typename Dtype>
cudaError_t CudaArray<Dtype>::device2host(Dtype* host_ptr) {
    cudaError_t err = cudaMemcpy(host_ptr, 
                                 __ptr, 
                                 __size * sizeof(Dtype), 
                                 cudaMemcpyDeviceToHost);
    return err;
}

template<typename Dtype>
cudaError_t CudaArray<Dtype>::host2device(const Dtype* host_ptr) {
    cudaError_t err = cudaMemcpy(__ptr, host_ptr,
                                 __size * sizeof(Dtype), 
                                 cudaMemcpyHostToDevice);
    return err;
}

template<typename Dtype>
CudaArray<Dtype>::CudaArray(const size_t size): __size(size) {

    cudaError_t err = cudaMalloc(&__ptr, __size*sizeof(Dtype));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

template<typename Dtype>
void CudaArray<Dtype>::set_all_zeros() {
    cudaError_t err = cudaMemset(__ptr, 0, __size*sizeof(Dtype));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

#endif

