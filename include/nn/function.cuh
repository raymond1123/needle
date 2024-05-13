#ifndef __FUNCTIONAL_CUH__
#define __FUNCTIONAL_CUH__

#include "tensor.cuh"
#include "ops/bp/padding.cuh"

template<typename Dtype>
using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

std::vector<int> ccc() {
    std::vector<int> out = {1,2,3,4,5};
    return out;
}

std::vector<int> ddd() {
    std::vector<int> out = {1};
    return out;
}

template<typename Dtype>
Tensor<Dtype> pad(Tensor<Dtype>& tensor, std::vector<int32_t> axes) {
    return tensor.padding(axes);
}

#endif

