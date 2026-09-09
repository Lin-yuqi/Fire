# ==============================
# Fire 核心库
# ==============================

aux_source_directory(base/ BASE_DIR)
aux_source_directory(op/ OP_DIR)
aux_source_directory(tensor/ TENSOR_DIR)
aux_source_directory(op/kernels/ KERNEL_DIR)
aux_source_directory(op/kernels/cpu/ CPU_KERNEL_DIR)
aux_source_directory(op/kernels/cuda/ CUDA_KERNEL_DIR)
aux_source_directory(model/ MODEL_DIR)

add_library(fire STATIC
    ${BASE_DIR}
    ${OP_DIR}
    ${TENSOR_DIR}
    ${MODEL_DIR}

    ${KERNEL_DIR}
    ${CPU_KERNEL_DIR}
    ${CUDA_KERNEL_DIR}
)


target_include_directories(fire
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${ARMADILLO_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/op/kernels
        /usr/local/cuda/targets/x86_64-linux/include/cccl
)

target_link_libraries(fire
    PUBLIC
        CUDA::cudart
        glog::glog
        ${ARMADILLO_LIBRARIES}
)

set_target_properties(fire PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_library(Fire::fire ALIAS fire)
