
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

add_test(
    NAME fire_export_tinyllama_scaffold
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_model/test_export_tinyllama.py
)
