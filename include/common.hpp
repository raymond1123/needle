#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include "log4cplus/logger.h"
#include "log4cplus/consoleappender.h"
#include "log4cplus/loglevel.h"
#include <log4cplus/loggingmacros.h>
#include <log4cplus/initializer.h>
#include <log4cplus/configurator.h>
#include <iomanip>

#include <string.h>
#include <math.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <tuple>
#include <utility> // pair
#include <unordered_map>
#include <map>
#include <set>

#include <cuda_runtime.h>
#include "helper_cuda.h"

#define CPU "cpu"
#define CUDA "cuda"

#define DISABLE_COPY_AND_ASSIGN(classname) \
private: \
    classname(const classname&); \
    classname& operator=(const classname&);

void Warning(const std::string &warning);

#endif

