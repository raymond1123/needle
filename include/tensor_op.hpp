#ifndef __TENOPS_HPP__
#define __TENOPS_HPP__

#include "ewise_add.cuh"
#include "ewise_minus.cuh"
#include "ewise_mul.cuh"
#include "ewise_div.cuh"

template<typename Dtype> class TensorOP;

template<typename Dtype> class EwiseAdd;
template<typename Dtype> class EwiseMinus;
template<typename Dtype> class EwiseMul;
template<typename Dtype> class EwiseDiv;

#endif

