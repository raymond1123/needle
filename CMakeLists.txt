cmake_minimum_required(VERSION 3.12)
project(needle LANGUAGES CXX CUDA)  # set project name

SET(CMAKE_CXX_STANDARD 17)  # set C++ standard to 11
SET(CMAKE_CUDA_ARCHITECTURES 86)

#set(CMAKE_BUILD_TYPE Release)  # enable debug symbols for gdb debugging
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") # 定义Release编译参数
SET(CMAKE_BUILD_TYPE Debug)  # enable debug symbols for gdb debugging
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb") # 定义Debug编译参数

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -arch compute_86 -code sm_86 # Specify your desired CUDA architecture
)

SET(PROJ_HEAD ${PROJECT_SOURCE_DIR}/include)
#SET(CUDA_HEAD "/usr/local/cuda-11.4/targets/x86_64-linux/include/")

FILE(GLOB_RECURSE TEST_SRC_FILES
    ${PROJECT_SOURCE_DIR}/unittest/test.cpp
    ${PROJECT_SOURCE_DIR}/lib/*.cpp
    ${PROJECT_SOURCE_DIR}/lib/*.cu)

FILE(GLOB_RECURSE PY_SRC_FILES
    ${PROJECT_SOURCE_DIR}/unittest/test.cpp
    ${PROJECT_SOURCE_DIR}/lib/*.cpp
    ${PROJECT_SOURCE_DIR}/lib/*.cu)

# create a library target
#add_library(mytensor SHARED ${TENSOR})
SET(LOG4CPP "/usr/local/lib/liblog4cplus.so")
#add_subdirectory(${PROJECT_SOURCE_DIR}/third_part/pybind11)

# find pybind
#execute_process(COMMAND python3 -m pybind11 --cmakedir
#RESULT_VARIABLE __pybind_exit_code
#OUTPUT_VARIABLE __pybind_path
#OUTPUT_STRIP_TRAILING_WHITESPACE)
#find_package(pybind11 PATHS ${__pybind_path})
#include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
#list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})
list(APPEND LINKER_LIBS ${LOG4CPP})
add_subdirectory("./third_part/pybind11")

####################
### CUDA BACKEND ###
####################
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Found cuda, building cuda backend")

  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} ${PROJ_HEAD})
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

  ## invoke nvidia smi to detect if we really have a GPU
  ##execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)
  #if(NV_RET EQUAL "0")
  #  CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  #else()
  #  # set to 3.7 the flag of K80
  #  CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 8.6)
  #endif()

  # set arch flags properly
  #CUDA_ADD_LIBRARY(unittest MODULE ${TEST_SRC_FILES} OPTIONS ${ARCH_FLAGS})

  pybind11_add_module(unittest ${TEST_SRC_FILES})
  target_link_libraries(unittest PUBLIC ${LINKER_LIBS})

  #pybind11_extension(unittest)
  #pybind11_strip(unittest)

  # directly output to ffi folder
  set_target_properties(unittest
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/unittest/
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden")


endif()

#include_directories(${PROJ_HEAD} ${CUDA_HEAD})
#pybind11_add_module(unittest ${PY_SRC_FILES})
#target_link_libraries(unittest PUBLIC ${LOG4CPP}
#cuda cudart cudnn)

#add_executable(unittest ${TEST_SRC_FILES})
#target_link_libraries(unittest PUBLIC ${LOG4CPP})
#target_link_libraries(libgfinfer PUBLIC ${LOG4CPP})


