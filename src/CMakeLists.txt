# ==============================
# Fire 核心库
# ==============================

aux_source_directory(base/ BASE_DIR)
aux_source_directory(op/ OP_DIR)
aux_source_directory(tensor/ TENSOR_DIR)

add_library(fire STATIC 
    ${BASE_DIR}
    ${OP_DIR}
    ${TENSOR_DIR}
)


target_include_directories(fire
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${ARMADILLO_INCLUDE_DIRS}
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
