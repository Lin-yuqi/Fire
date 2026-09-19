cmake_minimum_required(VERSION 3.17)

project(Fire LANGUAGES CXX CUDA)

# ==============================
# C++ 基本配置
# ==============================

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# 给 clangd / VSCode 使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ==============================
# 第三方依赖
# ==============================

# Some distributions (including Ubuntu 22.04) install SentencePiece headers
# and libraries without its CMake package configuration. Prefer the upstream
# package when available, then fall back to locating the installed artifacts.
find_package(sentencepiece CONFIG QUIET)

if(TARGET sentencepiece::sentencepiece)
    set(SENTENCEPIECE_TARGET sentencepiece::sentencepiece)
elseif(TARGET sentencepiece)
    set(SENTENCEPIECE_TARGET sentencepiece)
else()
    find_path(SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h)
    find_library(SENTENCEPIECE_LIBRARY NAMES sentencepiece)

    if(NOT SENTENCEPIECE_INCLUDE_DIR OR NOT SENTENCEPIECE_LIBRARY)
        message(FATAL_ERROR
            "SentencePiece headers or library were not found. "
            "Install the SentencePiece development package."
        )
    endif()

    add_library(SentencePiece::sentencepiece UNKNOWN IMPORTED)
    set_target_properties(SentencePiece::sentencepiece PROPERTIES
        IMPORTED_LOCATION "${SENTENCEPIECE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SENTENCEPIECE_INCLUDE_DIR}"
    )
    set(SENTENCEPIECE_TARGET SentencePiece::sentencepiece)
endif()
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)
find_package(CUDAToolkit REQUIRED)


# ==============================
# Fire 核心库
# ==============================

add_subdirectory(src)

# ==============================
# demo
# ==============================

add_subdirectory(demo)

# ==============================
# Tests
# ==============================
include(CTest)

if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    add_subdirectory(test)
endif()

