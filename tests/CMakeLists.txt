add_executable(test_storage test_storage.cpp)
target_link_libraries(test_storage PRIVATE vectordb_core)
add_test(NAME test_storage COMMAND test_storage)

add_executable(test_index test_index.cpp)
target_link_libraries(test_index PRIVATE vectordb_core)
add_test(NAME test_index COMMAND test_index)

add_executable(test_distance test_distance.cpp)
target_link_libraries(test_distance PRIVATE vectordb_core)
add_test(NAME test_distance COMMAND test_distance)
