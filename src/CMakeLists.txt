# ==============================
# Fire 核心库
# ==============================

aux_source_directory(base/ BASE_DIR)

add_library(fire STATIC 
    ${BASE_DIR}
)


target_include_directories(fire
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(fire 
    PRIVATE 
        CUDA::cudart)
        
target_link_libraries(fire
    PUBLIC
        glog::glog
        ${ARMADILLO_LIBRARIES}
)

set_target_properties(fire PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_library(Fire::fire ALIAS fire)