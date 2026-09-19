
aux_source_directory(test_op/ OP)
aux_source_directory(test_model MODEL)
aux_source_directory(test_tokenizer TOKEN)
aux_source_directory(test_sampler sampler)

add_executable(fire_tests
    test_base/test_buffer.cpp
    test_model/test_fire_reader.cpp
    test_tensor/test_tensor.cpp
    utils.cu
    ${OP}
    ${MODEL}
    ${TOKEN}
    ${sampler}
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
set(FIRE_TINYLLAMA_PATH "${CMAKE_SOURCE_DIR}/tmp/llama.fire")
set(FIRE_TINYLLAMA_TOKENIZER_PATH
    "${CMAKE_SOURCE_DIR}/models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model"
)

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

add_custom_target(fire_v1_fixture DEPENDS ${FIRE_V1_FIXTURE_PATH})
add_dependencies(fire_tests fire_v1_fixture)

target_compile_definitions(
    fire_tests
    PRIVATE FIRE_V1_FIXTURE_PATH="${FIRE_V1_FIXTURE_PATH}"
    PRIVATE FIRE_TINYLLAMA_PATH="${FIRE_TINYLLAMA_PATH}"
    PRIVATE FIRE_TINYLLAMA_TOKENIZER_PATH="${FIRE_TINYLLAMA_TOKENIZER_PATH}"
)

add_test(
    NAME fire_export_tinyllama_contract
    COMMAND
        ${Python3_EXECUTABLE}
        -B
        ${CMAKE_CURRENT_SOURCE_DIR}/test_model/test_export_tinyllama.py
)
