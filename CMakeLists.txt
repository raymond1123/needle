cmake_minimum_required(VERSION 3.12)
project(needle LANGUAGES CXX CUDA)  # set project name

# Set the C++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CUDA_ARCHITECTURES 86)
SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

SET(LOG4CPP "/usr/local/lib/liblog4cplus.so")

# Find CUDA package
find_package(CUDA REQUIRED)

# Include directories
#SET(CUDA_HEAD "/usr/local/cuda-11.4/targets/x86_64-linux/include/")
#include_directories(${CUDA_INCLUDE_DIRS}
#include_directories(${CUDA_HEAD} ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS} 
                    ${CMAKE_SOURCE_DIR} 
                    ${CMAKE_SOURCE_DIR}/include
                    ${CMAKE_SOURCE_DIR}/kernel/include)

message(STATUS "Found cuda, building cuda backend")
message(STATUS "cuda header: ${CUDA_INCLUDE_DIRS}")

# Add your source files
FILE(GLOB_RECURSE TEST_SRC_FILES
    ${CMAKE_SOURCE_DIR}/unit_test/test.cpp
    ${CMAKE_SOURCE_DIR}/lib/*.cpp
    ${CMAKE_SOURCE_DIR}/kernel/*.cu)

# pybind11
add_subdirectory(${CMAKE_SOURCE_DIR}/third_part/pybind11)
pybind11_add_module(Unittest ${TEST_SRC_FILES})

# Create an executable
#add_executable(unittest ${TEST_SRC_FILES})

# Link against the CUDA library
target_link_libraries(Unittest PUBLIC ${CUDA_LIBRARIES} ${LOG4CPP})

# directly output to ffi folder
set_target_properties(Unittest
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/unit_test/)

