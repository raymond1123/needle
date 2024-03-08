#ifndef __CUDA_TENSOR_HPP__
#define __CUDA_TENSOR_HPP__

#include "backend/cuda_backend.cuh"

namespace py = pybind11;

template<typename Dtype> class BaseTensor;

template<typename Dtype>
class CudaTensor: public BaseTensor<Dtype> {
public:
    explicit CudaTensor(py::array_t<Dtype>& np_array);
    ~CudaTensor() {}

    virtual py::array_t<Dtype> to_numpy() override;

protected:
    virtual void _from_numpy(py::array_t<Dtype> &a) override;

private:
    std::shared_ptr<CudaArray<Dtype>> __array;
};

//BaseTensor<Dtype, CudaTensor>(np_array, Backend::GPU),
template<typename Dtype>
CudaTensor<Dtype>::CudaTensor(py::array_t<Dtype>& np_array):
    BaseTensor<Dtype>(np_array) {
    //__array(std::make_shared<CudaArray<Dtype>>(this->_prod(this->__shape))) {
    size_t size = this->_prod(this->__shape);
    __array.reset(new CudaArray<Dtype>(size));
        std::cout << "selected cuda backend" << std::endl;
    _from_numpy(np_array);
}

template<typename Dtype>
void CudaTensor<Dtype>::_from_numpy(py::array_t<Dtype> &a) {
    cudaError_t err = __array->host2device(
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
    Dtype* host_ptr = (Dtype*)std::malloc(__array->size() * sizeof(Dtype));
    if (host_ptr == 0) throw std::bad_alloc();

    cudaError_t err = __array->device2host(host_ptr);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<Dtype>(this->__shape, numpy_strides, 
                              host_ptr + this->__offset, 
                              deallocate_buffer);
}

#endif

