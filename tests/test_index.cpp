#include <iostream>
#include <cassert>
#include "vectordb/FlatIndex.h"
#include "vectordb/HNSWIndex.h"

void test_flat_index() {
    std::cout << "[TEST] Running Flat Index Test..." << std::endl;

    vectordb::FlatIndex index;
    index.add_vector({1, {1.0f, 1.0f}});
    index.add_vector({2, {5.0f, 5.0f}});
    index.add_vector({3, {1.1f, 1.1f}});

    assert(index.size() == 3);

    auto results = index.search({1.0f, 1.0f}, 2);
    assert(results.size() == 2);
    assert(results[0].id == 1);
    assert(results[1].id == 3);

    // Deletion
    assert(index.remove_vector(1) == true);
    assert(index.size() == 2);
    auto results_after = index.search({1.0f, 1.0f}, 2);
    assert(results_after[0].id == 3);

    std::cout << "[PASS] Flat Index Test Passed!" << std::endl;
}

void test_hnsw_index() {
    std::cout << "[TEST] Running HNSW Index Test..." << std::endl;

    vectordb::HNSWIndex index(16, 200, 50);
    index.add_vector({1, {1.0f, 1.0f, 1.0f}});
    index.add_vector({2, {10.0f, 10.0f, 10.0f}});
    index.add_vector({3, {1.05f, 1.05f, 1.05f}});

    assert(index.size() == 3);

    auto results = index.search({1.0f, 1.0f, 1.0f}, 2);
    assert(results.size() == 2);
    assert(results[0].id == 1);
    assert(results[1].id == 3);

    std::cout << "[PASS] HNSW Index Test Passed!" << std::endl;
}

void test_hnsw_deletion_and_level_demotion() {
    std::cout << "[TEST] Running HNSW Vector Deletion & Level Demotion Test..." << std::endl;

    vectordb::HNSWIndex index(16, 200, 50);

    // Insert 20 vectors
    for (uint64_t i = 1; i <= 20; ++i) {
        float v = static_cast<float>(i);
        index.add_vector({i, {v, v, v}});
    }

    assert(index.size() == 20);

    // Delete vector 10
    assert(index.remove_vector(10) == true);
    assert(index.size() == 19);
    assert(index.remove_vector(10) == false); // Non-existent deletion fails

    // Search query around {10, 10, 10} - should not return ID 10
    auto results = index.search({10.0f, 10.0f, 10.0f}, 5);
    for (const auto& res : results) {
        assert(res.id != 10 && "Deleted vector ID 10 must not be returned in search");
    }

    // Delete ALL vectors to test total graph emptying and level demotion
    for (uint64_t i = 1; i <= 20; ++i) {
        index.remove_vector(i);
    }

    assert(index.size() == 0);
    auto empty_res = index.search({1.0f, 1.0f, 1.0f}, 2);
    assert(empty_res.empty() && "Search on emptied graph should return no results");

    std::cout << "[PASS] HNSW Deletion & Level Demotion Test Passed!" << std::endl;
}

int main() {
    test_flat_index();
    test_hnsw_index();
    test_hnsw_deletion_and_level_demotion();
    std::cout << "\n[ALL INDEX TESTS PASSED]" << std::endl;
    return 0;
}
