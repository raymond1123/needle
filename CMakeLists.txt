cmake_minimum_required(VERSION 3.12)
project(needle LANGUAGES CXX CUDA)  # set project name

# Set the C++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CUDA_ARCHITECTURES 86)
SET(CMAKE_BUILD_TYPE Debug)
#SET(CMAKE_BUILD_TYPE Release)  # enable debug symbols for gdb debugging
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

set(THIRD_PARTY "${PROJECT_SOURCE_DIR}/3rdparty")

set(CUTLASS_PATH ${THIRD_PARTY}/cutlass/build/install)
set(CUTLASS_LIB ${CUTLASS_PATH}/lib/libcutlass.so)
set(CUTLASS_HEAD ${CUTLASS_PATH}/include)

# Find CUDA package
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include
            ${CUTLASS_HEAD} ${CUDA_INCLUDE_DIRS})

message(STATUS "Found cuda, building cuda backend")
message(STATUS "cuda header: ${CUDA_INCLUDE_DIRS}")

# Add your source files
file(GLOB_RECURSE SRC_FILES
    ${PROJECT_SOURCE_DIR}/lib/tensor.cu)

# Set DEBUG to ON by default
option(DEBUG "Enable debug mode" ON)

# Add a definition if DEBUG is enabled
if (DEBUG)
    add_definitions(-DDEBUG)
endif()

# pybind11
add_subdirectory(${THIRD_PARTY}/pybind11)
pybind11_add_module(tensor ${SRC_FILES})

# Create an executable
#add_executable(unittest ${TEST_SRC_FILES})

# Link against the CUDA library
target_link_libraries(tensor PUBLIC ${CUDA_LIBRARIES})

# directly output to ffi folder
set_target_properties(tensor
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/unit_test/)

