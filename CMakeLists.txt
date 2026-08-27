cmake_minimum_required(VERSION 3.17)

project(Fire LANGUAGES CXX CUDA)

# ==============================
# C++ 基本配置
# ==============================

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# 给 clangd / VSCode 使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ==============================
# 第三方依赖
# ==============================

find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)


# ==============================
# Fire 核心库
# ==============================

add_subdirectory(src)


# ==============================
# Tests
# ==============================
include(CTest)

if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    add_subdirectory(test)
endif()


# ==============================
# Tools
# ==============================

add_subdirectory(tools)