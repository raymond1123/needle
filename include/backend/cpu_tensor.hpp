#ifndef __CPU_TENSOR_HPP__
#define __CPU_TENSOR_HPP__

#include "backend/base_tensor.hpp"
#include "backend/base_array.hpp"
#include "backend/cpu_backend.hpp"

namespace py = pybind11;

//template<typename Dtype> class BaseTensor;

template<typename Dtype>
class CpuTensor: public BaseTensor<Dtype> {
public:
    using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;
    explicit CpuTensor(const std::vector<int32_t>& shape, 
                       bool create_cache=true);
    CpuTensor(const std::shared_ptr<GenericOp<Dtype>> op, 
               std::vector<cached_data_type> inputs): BaseTensor<Dtype>(op, inputs) {}
    explicit CpuTensor(py::array_t<Dtype>& np_array);
    ~CpuTensor() {}

    CpuTensor(const CpuTensor&)=delete;
    CpuTensor& operator=(const CpuTensor&)=delete;

    virtual py::array_t<Dtype> to_numpy() override;
    virtual void fill_val(Dtype val) override;
    virtual void zeros() override;
    virtual void ones() override;
    virtual void from_buffer() override;

    inline virtual size_t size() override {
        return this->_prod(this->__shape);
    }
    virtual std::shared_ptr<BaseTensor<Dtype>> deep_cpy_cached_data() override;

    virtual inline BackendType device() override {return BackendType::CPU;} 

protected:
    virtual void _from_numpy(py::array_t<Dtype> &a) override;

/*
private:
    std::shared_ptr<CpuArray<Dtype>> array;
*/
};

template<typename Dtype>
CpuTensor<Dtype>::CpuTensor(py::array_t<Dtype>& np_array):
    BaseTensor<Dtype>(np_array) {
    size_t size = this->_prod(this->__shape);
    this->array.reset(new CpuArray<Dtype>(size));
    std::cout << "selected cpu backend" << std::endl;
    _from_numpy(np_array);
}

template<typename Dtype>
CpuTensor<Dtype>::CpuTensor(const std::vector<int32_t>& shape,
                            bool create_cache):
    BaseTensor<Dtype>(shape) {
    size_t size = this->_prod(this->__shape);
    if(create_cache)
        this->array.reset(new CpuArray<Dtype>(size));

    std::cout << "selected cpu backend" << std::endl;
}

template<typename Dtype>
void CpuTensor<Dtype>::fill_val(Dtype val) {
    this->array->fill_val(val);
}

template<typename Dtype>
void CpuTensor<Dtype>::zeros() {
    this->array->fill_val(static_cast<Dtype>(0));
}

template<typename Dtype>
void CpuTensor<Dtype>::ones() {
    this->array->fill_val(static_cast<Dtype>(1));
}

template<typename Dtype>
void CpuTensor<Dtype>::from_buffer() {

    printf("[");
    for(size_t i=0; i<size(); ++i)
        printf("%f,", this->array->get_ptr()[i]);

    printf("]\n");

}

template<typename Dtype>
void CpuTensor<Dtype>::_from_numpy(py::array_t<Dtype> &a) {
    const Dtype* ptr = reinterpret_cast<const Dtype*>(a.data());
    this->array->mem_cpy(const_cast<Dtype*>(ptr), 
                         MemCpyType::Host2Host);
}

template<typename Dtype>
py::array_t<Dtype> CpuTensor<Dtype>::to_numpy() {
    std::vector<int32_t> numpy_strides = this->__strides;
    std::transform(numpy_strides.begin(), 
                   numpy_strides.end(), 
                   numpy_strides.begin(),
                   [](int32_t& c) { return c * sizeof(Dtype); });

    return py::array_t<Dtype>(this->__shape, numpy_strides, 
                              this->cached_ptr() + this->__offset);
}

template<typename Dtype>
std::shared_ptr<BaseTensor<Dtype>> CpuTensor<Dtype>::deep_cpy_cached_data() {
    this->compact();
    std::shared_ptr<BaseTensor<Dtype>> cached_data = 
        std::make_shared<CpuTensor<Dtype>>(this->__shape);

    this->array->mem_cpy(cached_data->cached_ptr(), MemCpyType::Hosta2Hostb);

    return cached_data;
}

#endif

