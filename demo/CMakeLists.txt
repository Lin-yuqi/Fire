add_executable(llama_chat llama_chat.cpp)

target_link_libraries(llama_chat
    PRIVATE
        Fire::fire
)

target_compile_definitions(llama_chat
    PRIVATE
        FIRE_DEMO_MODEL_PATH="${PROJECT_SOURCE_DIR}/tmp/llama.fire"
        FIRE_DEMO_TOKENIZER_PATH="${PROJECT_SOURCE_DIR}/models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model"
)

add_executable(qwen_chat qwen_chat.cpp)
target_link_libraries(qwen_chat PRIVATE Fire::fire)
target_compile_definitions(qwen_chat
    PRIVATE
        FIRE_VERSION="${PROJECT_VERSION}"
        FIRE_QWEN3_0_6B_MODEL_PATH="${PROJECT_SOURCE_DIR}/tmp/qwen3-0.6b.fire"
        FIRE_QWEN3_0_6B_TOKENIZER_DIRECTORY="${PROJECT_SOURCE_DIR}/models/Qwen3-0.6B"
        FIRE_QWEN3_8B_MODEL_PATH="${PROJECT_SOURCE_DIR}/tmp/qwen3-8B-awq.fire"
        FIRE_QWEN3_8B_TOKENIZER_DIRECTORY="${PROJECT_SOURCE_DIR}/models/Qwen3-8B-AWQ"
)
