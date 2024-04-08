#ifndef __BASE_ARRAY_HPP__
#define __BASE_ARRAY_HPP__

#include "common.hpp"

template<typename Dtype>
class BaseArray {
public:
    BaseArray(const size_t size):__size(size), __ptr(nullptr) {};
    virtual ~BaseArray() {}

    BaseArray(const BaseArray&)=delete;
    BaseArray& operator=(const BaseArray&)=delete;

    inline size_t size() {return __size;}
    inline Dtype* get_ptr() {return __ptr;}
    inline void set_ptr(Dtype * ptr) {__ptr = ptr;}

    virtual void mem_cpy(Dtype* ptr, 
                         MemCpyType mem_cpy_type)=0;
    virtual void fill_val(Dtype val)=0;

protected:
    Dtype *__ptr;
    size_t __size;
};

#endif

