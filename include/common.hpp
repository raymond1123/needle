#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <filesystem>

#include <NvInfer.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdarg.h>

#include <stdexcept>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <iostream>

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <iterator>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

enum class BackendType: int {
    CPU = 0,
    CUDA = 1
};

enum class MemCpyType: int {
    Host2Host = 0,
    Hosta2Hostb = 1,
    Host2Dev = 2,
    Dev2Host = 3,
    Dev2Dev = 4
};

enum class OpType: int {
    EWAddTensor = 0,
    EWAddScalar = 1,
    EWMinusTensor = 2,
    EWMinusScalar = 3,
    EWMulTensor = 4,
    EWMulScalar = 5,
    EWDivTensor = 6,
    EWDivScalar = 7,
    EWPowTensor = 8,
    EWPowScalar = 9,
    MatMul = 10,
    Neg = 11,
    Log = 12,
    Exp = 13,
    Relu = 14,
    Tanh = 15,
    Reshape = 16,
    BroadcastTo = 17,
    Transpose = 18,
    Permute = 19,
    Summation = 20,
    Slice = 21,
    Flip = 22,
    Dilate = 23,
    Undilate = 24,
    Stack = 25,
    Split = 26,
    Conv = 27
};

enum class DataType: int{
    FLOAT=0, HALF=1, 
    INT8=2, INT32=3,
    BOOL=4, UINT8=5
};

#endif

