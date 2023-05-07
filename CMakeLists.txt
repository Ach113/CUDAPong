cmake_minimum_required(VERSION 3.12)
project(cuda_pong)

# Run Conan version command
message("conan --version")
execute_process(
    COMMAND conan --version
    RESULT_VARIABLE CONAN_RESULT
)
message("RESULT_VARIABLE = ${RESULT_VARIABLE}")
message("CONAN_RESULT = ${CONAN_RESULT}")

# Run Conan install command before conan_basic_setup
execute_process(
    COMMAND conan install ${CMAKE_SOURCE_DIR} --build=missing
    RESULT_VARIABLE CONAN_RESULT
)
message("RESULT_VARIABLE = ${RESULT_VARIABLE}")
message("CONAN_RESULT = ${CONAN_RESULT}")
if(CONAN_RESULT)
    message(FATAL_ERROR "conan install failed!")
endif()

add_subdirectory(example/zmq/py_cpp/simple/gym_client)
