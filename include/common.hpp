#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <filesystem>

#include <NvInfer.h>
#include <cuda_runtime.h>
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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

enum class BackendType: int {
    CPU = 0,
    CUDA = 1
};

enum class DataType: int{
    FLOAT=0, HALF=1, 
    INT8=2, INT32=3,
    BOOL=4, UINT8=5
};

#endif

