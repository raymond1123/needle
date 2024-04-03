#ifndef __CPU_TENSOR_HPP__
#define __CPU_TENSOR_HPP__

#include "backend/base_tensor.hpp"
#include "backend/cpu_backend.hpp"

namespace py = pybind11;

//template<typename Dtype> class BaseTensor;

template<typename Dtype>
class CpuTensor: public BaseTensor<Dtype> {
public:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;
    explicit CpuTensor(const std::vector<size_t>& shape);
    CpuTensor(const std::shared_ptr<GenericOp<Dtype>> op, 
               std::vector<cached_data_type> inputs): BaseTensor<Dtype>(op, inputs) {}
    explicit CpuTensor(py::array_t<Dtype>& np_array);
    ~CpuTensor() {}

    CpuTensor(const CpuTensor&)=delete;
    CpuTensor& operator=(const CpuTensor&)=delete;

    virtual py::array_t<Dtype> to_numpy() override;
    virtual void zeros() override;
    virtual void ones() override;

    inline virtual size_t size() override {return this->__array->size();}
    virtual inline Dtype* cached_ptr() override {
        return this->__array->get_ptr();
    }
    virtual std::shared_ptr<BaseTensor<Dtype>> deep_cpy_cached_data() const override;

    virtual inline BackendType device() override {return BackendType::CPU;} 

protected:
    virtual void _from_numpy(py::array_t<Dtype> &a) override;

private:
    std::shared_ptr<CpuArray<Dtype>> __array;
};

template<typename Dtype>
CpuTensor<Dtype>::CpuTensor(py::array_t<Dtype>& np_array):
    BaseTensor<Dtype>(np_array) {
    size_t size = this->_prod(this->__shape);
    __array.reset(new CpuArray<Dtype>(size));
    std::cout << "selected cpu backend" << std::endl;
    _from_numpy(np_array);
}

template<typename Dtype>
CpuTensor<Dtype>::CpuTensor(const std::vector<size_t>& shape):
    BaseTensor<Dtype>(shape) {
    size_t size = this->_prod(this->__shape);
    __array.reset(new CpuArray<Dtype>(size));
    std::cout << "selected cpu backend" << std::endl;
}

template<typename Dtype>
void CpuTensor<Dtype>::zeros() {
    this->__array->fill_val(static_cast<Dtype>(0));
}

template<typename Dtype>
void CpuTensor<Dtype>::ones() {
    this->__array->fill_val(static_cast<Dtype>(1));
}

template<typename Dtype>
void CpuTensor<Dtype>::_from_numpy(py::array_t<Dtype> &a) {
    __array->mem_cpy(reinterpret_cast<const Dtype*>(a.data()));
}

template<typename Dtype>
py::array_t<Dtype> CpuTensor<Dtype>::to_numpy() {
    std::vector<size_t> numpy_strides = this->__strides;
    std::transform(numpy_strides.begin(), 
                   numpy_strides.end(), 
                   numpy_strides.begin(),
                   [](size_t& c) { return c * sizeof(Dtype); });

    return __array->to_np(this->__shape, numpy_strides, this->__offset);
}

template<typename Dtype>
std::shared_ptr<BaseTensor<Dtype>> CpuTensor<Dtype>::deep_cpy_cached_data() const {
    std::shared_ptr<BaseTensor<Dtype>> cached_data = 
        std::make_shared<CpuTensor<Dtype>>(this->__shape);

    __array->deep_cpy(cached_data->cached_ptr());

    return cached_data;

}

#endif

