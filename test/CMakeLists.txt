
aux_source_directory(test_op/ test)

add_executable(fire_tests
    test_base/test_buffer.cpp
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