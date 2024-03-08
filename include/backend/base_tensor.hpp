#ifndef __BASE_TENSOR_HPP__
#define __BASE_TENSOR_HPP__

#include "common.hpp"
#include "backend/cpu_tensor.hpp"
#include "backend/cuda_tensor.hpp"

template<typename Dtype>
class BaseTensor {
public:
    BaseTensor(py::array_t<Dtype>& np_array);
    virtual ~BaseTensor()=default;

    virtual py::array_t<Dtype> to_numpy()=0;
    virtual std::vector<size_t> shape();

protected:
    virtual void _from_numpy(py::array_t<Dtype> &np_array)=0;

    size_t _prod(const std::vector<size_t> &shape);
    void _get_info_from_numpy(py::array_t<Dtype> &np_array);
    void _compact_strides();

protected:
    py::dtype __dtype;
    std::vector<size_t> __shape, __strides;
    size_t __offset;
};

template<typename Dtype>
BaseTensor<Dtype>::BaseTensor(py::array_t<Dtype>& np_array): __offset(0) {
    _get_info_from_numpy(np_array);
}

template<typename Dtype>
size_t BaseTensor<Dtype>::_prod(const std::vector<size_t>& shape) {
    size_t size = 1;
    for (auto &s: __shape)
        size *= s;
    return size;
}

template<typename Dtype>
void BaseTensor<Dtype>::_compact_strides() {
    std::vector<size_t> r_shape(__shape);
    reverse(r_shape.begin(), r_shape.end());

    __strides[__strides.size()-1] = 1;

    /* 
     * do not need to check whether strides.size()==1
     * the for loop does the job
     */
    for(int i=__strides.size()-2; i>=0; --i) {
        __strides[i] = __strides[i+1]*__shape[i+1];
    }
}

template<typename Dtype>
void BaseTensor<Dtype>::_get_info_from_numpy(py::array_t<Dtype>& np_array) {
    py::buffer_info buf_info = np_array.request();

    for (auto dim : buf_info.shape)
        __shape.push_back(dim);

    for (auto stride : buf_info.strides)
        __strides.push_back(stride / sizeof(Dtype));

    __dtype = np_array.dtype();

    if (__dtype.is(py::dtype::of<float>()))
        std::cout << "The array has float precision." << std::endl;
}

template<typename Dtype>
std::vector<size_t> BaseTensor<Dtype>::shape() {
    return __shape;
}

#endif

