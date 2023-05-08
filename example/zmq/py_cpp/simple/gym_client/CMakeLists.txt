cmake_minimum_required(VERSION 3.5)
project(cpp_pong_client)

# Use C++11
set(CMAKE_CXX_STANDARD 11)

find_package(PythonInterp 3.7)
find_package(PythonLibs 3.7)

add_subdirectory(pybind11)

# Add the source file
add_executable(${PROJECT_NAME} simple_pong_client.cpp)

# Find ZeroMQ package
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)
pkg_check_modules(GLIB2 REQUIRED gobject-2.0)
# pkg_check_modules(FFI REQUIRED ffi)

find_package(/usr/local/lib/cmake/opencv4/OpenCV)


# Link against the necessary libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
${OpenCV_LIBRARIES}
${PYTHON_LIBRARIES}
${ZMQ_LIBRARIES}
${GLIB2_LIBRARIES}
/usr/lib/x86_64-linux-gnu/libffi.a
# ${CONAN_TARGETS}
# webpdecoder
# CONAN_PKG::zmq
)

# message("CONAN_INCLUDE_DIRS = ${CONAN_INCLUDE_DIRS}")
target_include_directories(${PROJECT_NAME} PUBLIC
${OpenCV_INCLUDE_DIRS}
${PYTHON_INCLUDE_DIRS}
${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include
${ZMQ_INCLUDE_DIRS}
${GLIB2_INCLUDE_DIRS}
/usr/include/x86_64-linux-gnu/ # ffi.h
# ${CONAN_INCLUDE_DIRS}
)

# Set compile options and flags if needed
# target_compile_options(${PROJECT_NAME} PRIVATE ${CONAN_DEFINES})

# Set C++ standard if needed
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
