#ifndef __CPU_BACKEND_HPP__
#define __CPU_BACKEND_HPP__

#include "common.hpp"

#define ALIGNMENT 256
namespace py = pybind11;

template<typename Dtype>
class CpuArray: public BaseArray<Dtype> {
public:
    CpuArray(const size_t size);
    ~CpuArray() {free(this->__ptr);}

    CpuArray(const CpuArray&)=delete;
    CpuArray& operator=(const CpuArray&)=delete;

    void mem_cpy(Dtype* ptr,
                 MemCpyType mem_cpy_type) override;

    /*
    py::array_t<Dtype> to_np(std::vector<size_t> shape,
               std::vector<size_t> strides,
               size_t offset);
    */

    //inline size_t size() {return __size;}
    //inline Dtype* get_ptr() {return __ptr;}

    virtual void fill_val(Dtype val) override;

private:
    void __deep_cpy(Dtype* other_ptr);
/*
private:
    Dtype *__ptr;
    size_t __size;
*/
};

template<typename Dtype>
CpuArray<Dtype>::CpuArray(const size_t size): BaseArray<Dtype>(size) {

    int ret = posix_memalign(reinterpret_cast<void**>(&(this->__ptr)),
                             ALIGNMENT, this->__size*sizeof(Dtype));

    if (ret != 0) throw std::bad_alloc();
}

template<typename Dtype>
void CpuArray<Dtype>::mem_cpy(Dtype* ptr, MemCpyType mem_cpy_type) {
    if(mem_cpy_type==MemCpyType::Host2Host)
        std::memcpy(this->__ptr, ptr, this->__size*sizeof(Dtype));
    else if(mem_cpy_type==MemCpyType::Hosta2Hostb)
        __deep_cpy(ptr);
}

template<typename Dtype>
void CpuArray<Dtype>::__deep_cpy(Dtype* other_ptr) {
    for(size_t i=0; i<this->__size; ++i)
        other_ptr[i] = this->__ptr[i];
}

/*
template<typename Dtype>
py::array_t<Dtype> CpuArray<Dtype>::to_np(std::vector<size_t> shape,
                                   std::vector<size_t> strides,
                                   size_t offset) {
    return py::array_t<Dtype>(shape, strides, this->__ptr + offset);
}
*/

template<typename Dtype>
void CpuArray<Dtype>::fill_val(Dtype val) {
    if(val==static_cast<Dtype>(0)) {
        memset(this->__ptr, val, this->__size*sizeof(Dtype));
    } else {
        for(size_t i=0; i<this->__size; ++i)
            this->__ptr[i] = val;
    }
}

#endif

