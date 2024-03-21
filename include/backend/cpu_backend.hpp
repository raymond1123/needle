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

    void mem_cpy(const Dtype* host_ptr);
    py::array_t<Dtype> to_np(std::vector<size_t> shape,
               std::vector<size_t> strides,
               size_t offset);
    inline size_t size() {return __size;}
    inline Dtype* get_ptr() {return __ptr;}

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
py::array_t<Dtype> CpuArray<Dtype>::to_np(std::vector<size_t> shape,
                                   std::vector<size_t> strides,
                                   size_t offset) {
    return py::array_t<Dtype>(shape, strides, __ptr + offset);
}

#endif

