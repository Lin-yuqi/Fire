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
