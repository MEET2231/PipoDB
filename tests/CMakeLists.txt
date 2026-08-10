add_executable(test_storage test_storage.cpp)
target_link_libraries(test_storage PRIVATE vectordb_core)
add_test(NAME test_storage COMMAND test_storage)

add_executable(test_index test_index.cpp)
target_link_libraries(test_index PRIVATE vectordb_core)
add_test(NAME test_index COMMAND test_index)

add_executable(test_distance test_distance.cpp)
target_link_libraries(test_distance PRIVATE vectordb_core)
add_test(NAME test_distance COMMAND test_distance)

add_executable(test_collection test_collection.cpp)
target_link_libraries(test_collection PRIVATE vectordb_core)
add_test(NAME test_collection COMMAND test_collection)

add_executable(test_db test_db.cpp)
target_link_libraries(test_db PRIVATE vectordb_core)
add_test(NAME test_db COMMAND test_db)

add_executable(test_filter test_filter.cpp)
target_link_libraries(test_filter PRIVATE vectordb_core)
add_test(NAME test_filter COMMAND test_filter)

add_executable(benchmark_distance benchmark_distance.cpp)
target_link_libraries(benchmark_distance PRIVATE vectordb_core)
