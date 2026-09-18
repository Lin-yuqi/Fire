add_executable(llama_chat llama_chat.cpp)


target_link_libraries(llama_chat
    PRIVATE 
        Fire::fire
)