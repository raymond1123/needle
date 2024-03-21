#ifndef __CUDA_TENSOR_HPP__
#define __CUDA_TENSOR_HPP__

#include "backend/base_tensor.hpp"
#include "backend/cuda_backend.cuh"

namespace py = pybind11;

template<typename Dtype> class BaseTensor;

template<typename Dtype>
class CudaTensor: public BaseTensor<Dtype> {
public:
    explicit CudaTensor(py::array_t<Dtype>& np_array);
    explicit CudaTensor(const std::vector<size_t>& shape);
    ~CudaTensor() {}

    virtual py::array_t<Dtype> to_numpy() override;
    virtual void zeros() override;

    inline virtual size_t size() override {return this->__array->size();}
    virtual inline Dtype* cached_ptr() {
        return this->__array->get_ptr();
    }
    virtual inline BackendType device() override {return BackendType::CUDA;}

protected:
    virtual void _from_numpy(py::array_t<Dtype> &a) override;

private:
    std::shared_ptr<CudaArray<Dtype>> __array;
};

template<typename Dtype>
CudaTensor<Dtype>::CudaTensor(py::array_t<Dtype>& np_array):
    BaseTensor<Dtype>(np_array) {
    size_t size = this->_prod(this->__shape);
    this->__array.reset(new CudaArray<Dtype>(size));
    std::cout << "selected cuda backend" << std::endl;
    _from_numpy(np_array);
}

template<typename Dtype>
CudaTensor<Dtype>::CudaTensor(const std::vector<size_t>& shape):
    BaseTensor<Dtype>(shape) {
    size_t size = this->_prod(this->__shape);
    this->__array.reset(new CudaArray<Dtype>(size));
    std::cout << "selected cuda backend" << std::endl;
}

template<typename Dtype>
void CudaTensor<Dtype>::zeros() {
    this->__array->set_all_zeros();
}

template<typename Dtype>
void CudaTensor<Dtype>::_from_numpy(py::array_t<Dtype> &a) {
    cudaError_t err = this->__array->host2device(
            reinterpret_cast<const Dtype*>(a.data())
    );

    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

template<typename Dtype>
py::array_t<Dtype> CudaTensor<Dtype>::to_numpy() {

    std::vector<size_t> numpy_strides = this->__strides;
    std::transform(numpy_strides.begin(), 
                   numpy_strides.end(), 
                   numpy_strides.begin(),
                   [](size_t& c) { return c * sizeof(Dtype); });

    // copy memory to host
    Dtype* host_ptr = (Dtype*)std::malloc(this->__array->size() * sizeof(Dtype));
    if (host_ptr == 0) throw std::bad_alloc();

    cudaError_t err = this->__array->device2host(host_ptr);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<Dtype>(this->__shape, numpy_strides, 
                              host_ptr + this->__offset, 
                              deallocate_buffer);
}

#endif

