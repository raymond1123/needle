#ifndef __CUDA_BACKEND_HPP__
#define __CUDA_BACKEND_HPP__

#include "common.hpp"

template<typename Dtype>
__global__ void FillKernel(Dtype* out, Dtype val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

template<typename Dtype>
class CudaArray {
public:
    CudaArray(const size_t size);
    ~CudaArray() {
        cudaFree(__ptr);
        __size = 0;
    }

    CudaArray(const CudaArray&)=delete;
    CudaArray& operator=(const CudaArray&)=delete;

    inline size_t size() {return __size;}
    inline Dtype* get_ptr() {return __ptr;}
    cudaError_t device2host(Dtype* host_ptr);
    cudaError_t host2device(const Dtype* host_ptr);
    cudaError_t device2device(Dtype* other_device_ptr);

    void fill_val(Dtype val);

private:
    Dtype *__ptr;
    size_t __size;
};

template<typename Dtype>
cudaError_t CudaArray<Dtype>::device2host(Dtype* host_ptr) {
    cudaError_t err = cudaMemcpy(host_ptr, __ptr, 
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
cudaError_t CudaArray<Dtype>::device2device(Dtype* other_device_ptr) {
    cudaError_t err = cudaMemcpy(other_device_ptr, __ptr, 
                                 __size * sizeof(Dtype), 
                                 cudaMemcpyDeviceToDevice);
    return err;
}

template<typename Dtype>
CudaArray<Dtype>::CudaArray(const size_t size): __size(size) {

    cudaError_t err = cudaMalloc(&__ptr, __size*sizeof(Dtype));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

template<typename Dtype>
void CudaArray<Dtype>::fill_val(Dtype val) {
    if(val==static_cast<Dtype>(0)) {
        cudaError_t err = cudaMemset(__ptr, val, __size*sizeof(Dtype));
        if (err != cudaSuccess) 
            throw std::runtime_error(cudaGetErrorString(err));
    } else {
        int block = 256;
        int grid = (__size+block-1)/block;
        FillKernel<<<grid, block>>>(__ptr, val, __size);
    }
}

#endif

