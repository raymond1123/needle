#ifndef __CUDA_TENSOR_HPP__
#define __CUDA_TENSOR_HPP__

#include "backend/base_tensor.hpp"
#include "backend/base_array.hpp"
#include "backend/cuda_backend.cuh"

namespace py = pybind11;

template<typename Dtype> class BaseTensor;

template<typename Dtype>
class CudaTensor: public BaseTensor<Dtype> {
public:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;
    explicit CudaTensor(py::array_t<Dtype>& np_array);
    CudaTensor(const std::shared_ptr<GenericOp<Dtype>> op, 
               std::vector<cached_data_type> inputs): BaseTensor<Dtype>(op, inputs) {}
    explicit CudaTensor(const std::vector<size_t>& shape, 
                        bool create_cache=true);
    ~CudaTensor() {}

    CudaTensor(const CudaTensor&)=delete;
    CudaTensor& operator=(const CudaTensor&)=delete;

    virtual py::array_t<Dtype> to_numpy() override;
    virtual void zeros() override;
    virtual void ones() override;

    inline virtual size_t size() override {
        return this->_prod(this->__shape);
    }
    virtual std::shared_ptr<BaseTensor<Dtype>> deep_cpy_cached_data() const override;
    virtual inline BackendType device() override {return BackendType::CUDA;}

protected:
    virtual void _from_numpy(py::array_t<Dtype> &a) override;

/*
private:
    std::shared_ptr<CudaArray<Dtype>> array;
*/
};

template<typename Dtype>
CudaTensor<Dtype>::CudaTensor(py::array_t<Dtype>& np_array):
    BaseTensor<Dtype>(np_array) {
    size_t size = this->_prod(this->__shape);
    this->array.reset(new CudaArray<Dtype>(size));
    std::cout << "selected cuda backend 1" << std::endl;
    _from_numpy(np_array);
}

template<typename Dtype>
CudaTensor<Dtype>::CudaTensor(const std::vector<size_t>& shape, 
                              bool create_cache):
    BaseTensor<Dtype>(shape) {
    size_t size = this->_prod(this->__shape);
    this->array.reset(new CudaArray<Dtype>(size, create_cache));

    std::cout << "selected cuda backend 2, " << create_cache << std::endl;
}

template<typename Dtype>
void CudaTensor<Dtype>::zeros() {
    this->array->fill_val(static_cast<Dtype>(0));
    this->is_compact = true;
    this->cached = true;
}

template<typename Dtype>
void CudaTensor<Dtype>::ones() {
    this->array->fill_val(static_cast<Dtype>(1));
    this->is_compact = true;
    this->cached = true;
}

template<typename Dtype>
void CudaTensor<Dtype>::_from_numpy(py::array_t<Dtype> &a) {
    //cudaError_t err = this->array->host2device(
    const Dtype* ptr = reinterpret_cast<const Dtype*>(a.data());
    this->array->mem_cpy(const_cast<Dtype*>(ptr),
                         MemCpyType::Host2Dev);

}

template<typename Dtype>
py::array_t<Dtype> CudaTensor<Dtype>::to_numpy() {

    std::vector<size_t> numpy_strides = this->__strides;
    std::transform(numpy_strides.begin(), 
                   numpy_strides.end(), 
                   numpy_strides.begin(),
                   [](size_t& c) { return c * sizeof(Dtype); });

    // copy memory to host
    Dtype* host_ptr = (Dtype*)std::malloc(this->array->size() * sizeof(Dtype));
    if (host_ptr == 0) throw std::bad_alloc();

    this->array->mem_cpy(host_ptr, MemCpyType::Dev2Host);

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<Dtype>(this->__shape, numpy_strides, 
                              host_ptr + this->__offset, 
                              deallocate_buffer);
}

template<typename Dtype>
std::shared_ptr<BaseTensor<Dtype>> CudaTensor<Dtype>::deep_cpy_cached_data() const {
    std::shared_ptr<BaseTensor<Dtype>> cached_data = 
        std::make_shared<CudaTensor<Dtype>>(this->__shape);

    this->array->mem_cpy(cached_data->cached_ptr(), MemCpyType::Dev2Dev);

    return cached_data;
}

#endif

