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

int main() {
    test_flat_index();
    test_hnsw_index();
    std::cout << "\n[ALL INDEX TESTS PASSED]" << std::endl;
    return 0;
}
