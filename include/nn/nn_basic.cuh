#ifndef __NN_BASIC_CUH__
#define __NN_BASIC_CUH__

#include "tensor.cuh"
#include "ops/bp/padding.cuh"

template<typename Dtype>
using cached_data_type = std::shared_ptr<BaseTensor<Dtype>>;

/* special Tensor that represents parameters */
template<typename Dtype>
class Parameter: public Tensor<Dtype> {
};

#endif

