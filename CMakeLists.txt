cmake_minimum_required(VERSION 3.5)
project(cuda_pong)

# Run Conan version command
# message("conan --version")
# execute_process(
#     COMMAND conan --version
#     RESULT_VARIABLE CONAN_RESULT
# )
# message("RESULT_VARIABLE = ${RESULT_VARIABLE}")
# message("CONAN_RESULT = ${CONAN_RESULT}")

# Run Conan install command before conan_basic_setup
# execute_process(
#     COMMAND conan install ${CMAKE_SOURCE_DIR} --build=missing
#     RESULT_VARIABLE CONAN_RESULT
# )
# message("RESULT_VARIABLE = ${RESULT_VARIABLE}")
# message("CONAN_RESULT = ${CONAN_RESULT}")
# if(CONAN_RESULT)
#     message(FATAL_ERROR "conan install failed!")
# endif()

# Include Conan-generated CMake file
# include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
# conan_basic_setup(TARGETS)

# function(filter_conan_lib_dirs LIB_DIRS)
#     set(LIB_NAME "libwebp")  # Replace with the actual library name
#     list(FILTER CONAN_LIB_DIRS EXCLUDE REGEX "/${LIB_NAME}[^/]*$")
#     list(APPEND ${LIB_DIRS} ${CONAN_LIB_DIRS})
# endfunction()

# set(FILTERED_LIB_DIRS)
# filter_conan_lib_dirs(FILTERED_LIB_DIRS)
# message("FILTERED_LIB_DIRS = ${FILTERED_LIB_DIRS}")
# link_directories(${FILTERED_LIB_DIRS})

# add_subdirectory(example/zmq/py_cpp/simple/gym_client)
add_subdirectory(example/zmq/py_cpp/simple_v2/gym_client)
