cmake_minimum_required(VERSION 3.10)
project(VectorDB VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable maximum performance optimizations on Linux
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -pthread")

# Include header directory
include_directories(include)

# Compile source files into a core library
add_library(vectordb_core
    src/Distance.cpp
    src/FlatIndex.cpp
    src/Storage.cpp
    src/HNSWIndex.cpp
    src/Collection.cpp
    src/DB.cpp
    src/Filter.cpp
    src/SQ8Index.cpp
    src/Batch.cpp
)

# Build app executable
add_executable(vectordb_app app/main.cpp)
target_link_libraries(vectordb_app PRIVATE vectordb_core)

# Build network server executable
add_executable(pipodb_server app/pipodb_server.cpp)
target_link_libraries(pipodb_server PRIVATE vectordb_core)

enable_testing()
add_subdirectory(tests)