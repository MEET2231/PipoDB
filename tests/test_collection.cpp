#include <iostream>
#include <cassert>
#include <cstdio>
#include <sys/stat.h>
#include "vectordb/Collection.h"

void test_collection_operations() {
    std::cout << "[TEST] Running Collection Operations Test..." << std::endl;

    vectordb::CollectionParams params;
    params.name = "test_collection";
    params.dimension = 4;
    params.index_type = "HNSW";
    params.M = 16;

    vectordb::Collection collection(params);
    assert(collection.size() == 0);

    collection.add_vector(1, {1.0f, 0.0f, 0.0f, 0.0f}, "{\"title\": \"Vector 1\"}");
    collection.add_vector(2, {0.0f, 1.0f, 0.0f, 0.0f}, "{\"title\": \"Vector 2\"}");
    collection.add_vector(3, {1.1f, 0.1f, 0.0f, 0.0f}, "{\"title\": \"Vector 3\"}");

    assert(collection.size() == 3);

    // Search query
    auto hits = collection.search({1.0f, 0.0f, 0.0f, 0.0f}, 2, true);
    assert(hits.size() == 2);
    assert(hits[0].id == 1);
    assert(hits[0].payload_json == "{\"title\": \"Vector 1\"}");
    assert(hits[1].id == 3);

    std::cout << "[PASS] Collection Operations Test Passed!" << std::endl;
}

void test_collection_persistence() {
    std::cout << "[TEST] Running Collection Persistence Test..." << std::endl;

    std::string test_dir = "temp_col_dir";

    vectordb::CollectionParams params;
    params.name = "persist_col";
    params.dimension = 3;
    params.index_type = "HNSW";

    vectordb::Collection original(params);
    original.add_vector(101, {1.0f, 2.0f, 3.0f}, "{\"meta\": \"abc\"}");
    original.add_vector(102, {4.0f, 5.0f, 6.0f}, "{\"meta\": \"xyz\"}");

    assert(original.save_to_dir(test_dir) && "Failed to save collection");

    vectordb::Collection loaded(params);
    assert(loaded.load_from_dir(test_dir) && "Failed to load collection");

    assert(loaded.size() == 2);
    auto hits = loaded.search({1.0f, 2.0f, 3.0f}, 1, true);
    assert(!hits.empty());
    assert(hits[0].id == 101);
    assert(hits[0].payload_json == "{\"meta\": \"abc\"}");

    std::cout << "[PASS] Collection Persistence Test Passed!" << std::endl;
}

int main() {
    test_collection_operations();
    test_collection_persistence();
    std::cout << "\n[ALL COLLECTION TESTS PASSED]" << std::endl;
    return 0;
}
