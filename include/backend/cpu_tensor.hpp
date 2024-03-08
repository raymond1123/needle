#ifndef __CPU_TENSOR_HPP__
#define __CPU_TENSOR_HPP__

#include "backend/cpu_backend.hpp"

namespace py = pybind11;

template<typename Dtype> class BaseTensor;

template<typename Dtype>
class CpuTensor: public BaseTensor<Dtype> {
public:
    explicit CpuTensor(py::array_t<Dtype>& np_array);
    ~CpuTensor() {}

    virtual py::array_t<Dtype> to_numpy() override;

protected:
    virtual void _from_numpy(py::array_t<Dtype> &a) override;

private:
    std::shared_ptr<CpuArray<Dtype>> __array;
};

template<typename Dtype>
CpuTensor<Dtype>::CpuTensor(py::array_t<Dtype>& np_array):
    BaseTensor<Dtype>(np_array) {
    //__array(std::make_shared<CpuArray<Dtype>>(this->_prod(this->__shape))) {
    size_t size = this->_prod(this->__shape);
    __array.reset(new CpuArray<Dtype>(size));
    std::cout << "selected cpu backend" << std::endl;
    _from_numpy(np_array);
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

#endif

