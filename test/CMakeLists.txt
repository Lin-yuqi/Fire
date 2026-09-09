
aux_source_directory(test_op/ test)

add_executable(fire_tests
    test_base/test_buffer.cpp
    test_model/test_fire_reader.cpp
    test_tensor/test_tensor.cpp
    utils.cu
    ${test}
)

target_link_libraries(fire_tests
    PRIVATE
        Fire::fire
        GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(fire_tests)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(FIRE_V1_FIXTURE_PATH "${CMAKE_CURRENT_BINARY_DIR}/fixtures/fire_v1_contract.fire")

add_custom_command(
    OUTPUT ${FIRE_V1_FIXTURE_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/fixtures
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_model/generate_fire_v1_fixture.py
        ${FIRE_V1_FIXTURE_PATH}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_model/generate_fire_v1_fixture.py
    COMMENT "Generating independent .fire v1 contract fixture"
    VERBATIM
)

# Kept out of the default build until the fixture encoder and Reader are
# implemented. The target already fixes the future test's build-tree path.
add_custom_target(fire_v1_fixture DEPENDS ${FIRE_V1_FIXTURE_PATH})

target_compile_definitions(
    fire_tests
    PRIVATE FIRE_V1_FIXTURE_PATH="${FIRE_V1_FIXTURE_PATH}"
)

add_test(
    NAME fire_export_tinyllama_scaffold
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_model/test_export_tinyllama.py
)
