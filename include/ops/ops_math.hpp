#ifndef __OPS_MATH_OP__
#define __OPS_MATH_OP__

#include "backend/cpu_tensor.hpp"
#include "backend/cuda_tensor.hpp"
#include "ops/bp/ew.cuh"
#include "ops/bp/reshape.cuh"
#include "ops/bp/broadcast.cuh"
#include "ops/bp/permute.cuh"
#include "ops/bp/transpose.cuh"
#include "ops/bp/summation.cuh"
#include "ops/bp/max.cuh"
#include "ops/bp/slice.cuh"
#include "ops/bp/flip.cuh"
#include "ops/bp/matmul.cuh"
#include "ops/bp/setitem.cuh"
#include "ops/bp/padding.cuh"
#include "ops/bp/stack.cuh"
#include "ops/bp/split.cuh"
#include "ops/bp/dilate.cuh"
#include "ops/bp/relu.cuh"
#include "ops/bp/tanh.cuh"
#include "ops/bp/log.cuh"
#include "ops/bp/exp.cuh"
#include "ops/bp/neg.cuh"

#endif
