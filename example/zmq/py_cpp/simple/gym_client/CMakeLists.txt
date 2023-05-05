cmake_minimum_required(VERSION 3.5)
project(cpp_pong_client)

# Use C++11
set(CMAKE_CXX_STANDARD 11)

# Find and include ZMQ and OpenCV using Conan
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

# Add the source file
add_executable(${PROJECT_NAME} simple_pong_client.cpp)

# Link against the necessary libraries
target_link_libraries(${PROJECT_NAME} 
CONAN_PKG::opencv
CONAN_PKG::zmq
)