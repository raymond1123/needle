#ifndef __CPU_BACKEND_HPP__
#define __CPU_BACKEND_HPP__

#include "common.hpp"

#define ALIGNMENT 256
namespace py = pybind11;

template<typename Dtype>
class CpuArray {
public:
    CpuArray(const size_t size);
    ~CpuArray() {free(__ptr);}

    CpuArray(const CpuArray&)=delete;
    CpuArray& operator=(const CpuArray&)=delete;

    void mem_cpy(const Dtype* host_ptr);
    void deep_cpy(Dtype* other_ptr);
    py::array_t<Dtype> to_np(std::vector<size_t> shape,
               std::vector<size_t> strides,
               size_t offset);
    inline size_t size() {return __size;}
    inline Dtype* get_ptr() {return __ptr;}

    void fill_val(Dtype val);

private:
    Dtype *__ptr;
    size_t __size;
};

template<typename Dtype>
CpuArray<Dtype>::CpuArray(const size_t size): __size(size) {

    int ret = posix_memalign(reinterpret_cast<void**>(&__ptr), 
                             ALIGNMENT, __size*sizeof(Dtype));

    if (ret != 0) throw std::bad_alloc();
}

template<typename Dtype>
void CpuArray<Dtype>::mem_cpy(const Dtype* numpy_ptr) {
    std::memcpy(__ptr, numpy_ptr, __size*sizeof(Dtype));
}

template<typename Dtype>
void CpuArray<Dtype>::deep_cpy(Dtype* other_ptr) {
    for(size_t i=0; i<__size; ++i)
        other_ptr[i] = __ptr[i];
}

template<typename Dtype>
py::array_t<Dtype> CpuArray<Dtype>::to_np(std::vector<size_t> shape,
                                   std::vector<size_t> strides,
                                   size_t offset) {
    return py::array_t<Dtype>(shape, strides, __ptr + offset);
}

template<typename Dtype>
void CpuArray<Dtype>::fill_val(Dtype val) {
    if(val==static_cast<Dtype>(0)) {
        memset(__ptr, val, __size*sizeof(Dtype));
    } else {
        for(size_t i=0; i<__size; ++i)
            __ptr[i] = val;
    }
}

#endif

