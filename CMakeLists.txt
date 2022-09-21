cmake_minimum_required(VERSION 3.10)
project(gpu-camera)

set(CMAKE_CXX_STANDARD 17)
set(PROJECT_SOURCE_DIR src)

find_package(OpenCV REQUIRED)

#include_directories( ${OpenCV_INCLUDE_DIRS}  include )
add_compile_options(-Ofast)
include_directories(include)

file(GLOB SOURCES
    ${PROJECT_SOURCE_DIR}/*.h
    ${PROJECT_SOURCE_DIR}/*.cpp
    ${PROJECT_SOURCE_DIR}/*.hpp
)

# setup CUDA
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_53 SM_62")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62)

    if(CUDA_VERSION_MAJOR GREATER 9)
        message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_72")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_72,code=sm_72)
    endif()

    if(CUDA_VERSION_MAJOR GREATER 10)
        message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_87")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_87,code=sm_87)
    endif()
endif()

include_directories(/usr/local/cuda-10.0/targets/aarch64-linux/include/ /usr/include/gstreamer-1.0 /usr/include/glib-2.0 /usr/lib/aarch64-linux-gnu/glib-2.0/include)

add_executable(cuda-cam ${PROJECT_SOURCE_DIR}/cuda_cam.cpp)
target_link_libraries(cuda-cam ${OpenCV_LIBS} jetson-utils cuda cudart)
