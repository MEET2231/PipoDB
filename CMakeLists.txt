cmake_minimum_required(VERSION 3.10)
project(VectorDB VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable maximum performance optimizations on Linux
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Include header directory
include_directories(include)

# Compile source files into a core library
add_library(vectordb_core
    src/Distance.cpp
    src/FlatIndex.cpp
    src/Storage.cpp
    src/HNSWIndex.cpp
)

# Build executable and link against vectordb_core
add_executable(vectordb_app app/main.cpp)
target_link_libraries(vectordb_app PRIVATE vectordb_core)