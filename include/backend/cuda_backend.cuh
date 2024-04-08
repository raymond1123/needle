#ifndef __CUDA_BACKEND_HPP__
#define __CUDA_BACKEND_HPP__

#include "common.hpp"

template<typename Dtype>
__global__ void FillKernel(Dtype* out, Dtype val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

template<typename Dtype>
class CudaArray:public BaseArray<Dtype> {
public:
    CudaArray(const size_t size);
    ~CudaArray() {
        cudaFree(this->__ptr);
        this->__size = 0;
    }

    CudaArray(const CudaArray&)=delete;
    CudaArray& operator=(const CudaArray&)=delete;

    //inline size_t size() {return __size;}
    //inline Dtype* get_ptr() {return __ptr;}

    virtual void mem_cpy(Dtype* ptr, 
                         MemCpyType mem_cpy_type) override;

private:
    cudaError_t __device2host(Dtype* host_ptr);
    cudaError_t __host2device(const Dtype* host_ptr);
    cudaError_t __device2device(Dtype* other_device_ptr);

    virtual void fill_val(Dtype val) override;

/*
    Dtype *__ptr;
    size_t __size;
*/
};

template<typename Dtype>
CudaArray<Dtype>::CudaArray(const size_t size): BaseArray<Dtype>(size) {

    cudaError_t err = cudaMalloc(&this->__ptr, this->__size*sizeof(Dtype));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

template<typename Dtype>
void CudaArray<Dtype>::mem_cpy(Dtype* ptr, 
                               MemCpyType mem_cpy_type) {
    cudaError_t err = cudaSuccess;
    if (mem_cpy_type==MemCpyType::Host2Dev) {
        err = __host2device(ptr);
    } else if(mem_cpy_type==MemCpyType::Dev2Host) {
        err = __device2host(ptr);
    } else if(mem_cpy_type==MemCpyType::Dev2Dev) {
        err = __device2device(ptr);
    }

    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

template<typename Dtype>
cudaError_t CudaArray<Dtype>::__device2host(Dtype* host_ptr) {
    cudaError_t err = cudaMemcpy(host_ptr, this->__ptr, 
                                 this->__size * sizeof(Dtype), 
                                 cudaMemcpyDeviceToHost);
    return err;
}

template<typename Dtype>
cudaError_t CudaArray<Dtype>::__host2device(const Dtype* host_ptr) {
    cudaError_t err = cudaMemcpy(this->__ptr, host_ptr,
                                 this->__size * sizeof(Dtype), 
                                 cudaMemcpyHostToDevice);
    return err;
}

template<typename Dtype>
cudaError_t CudaArray<Dtype>::__device2device(Dtype* other_device_ptr) {
    cudaError_t err = cudaMemcpy(other_device_ptr, this->__ptr, 
                                 this->__size * sizeof(Dtype), 
                                 cudaMemcpyDeviceToDevice);
    return err;
}

template<typename Dtype>
void CudaArray<Dtype>::fill_val(Dtype val) {
    if(val==static_cast<Dtype>(0)) {
        cudaError_t err = cudaMemset(this->__ptr, val, 
                                     this->__size*sizeof(Dtype));
        if (err != cudaSuccess) 
            throw std::runtime_error(cudaGetErrorString(err));
    } else {
        int block = 256;
        int grid = (this->__size+block-1)/block;
        FillKernel<<<grid, block>>>(this->__ptr, val, this->__size);
    }
}

#endif

