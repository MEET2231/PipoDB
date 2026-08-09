#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <cstdio>
#include "vectordb/HNSWIndex.h"
#include "vectordb/FlatIndex.h"
#include "vectordb/Storage.h"

void test_hnsw_persistence() {
    std::cout << "[TEST] Running HNSW Index Persistence Test..." << std::endl;

    const std::string filename = "temp_test_hnsw.bin";
    std::remove(filename.c_str());

    // 1. Build original HNSW index
    vectordb::HNSWIndex original_index(16, 200, 50);

    for (uint64_t i = 1; i <= 50; ++i) {
        float val = static_cast<float>(i);
        std::vector<float> vec_data = {val, val * 2.0f, val * 3.0f, val * 0.5f};
        original_index.add_vector({i, vec_data});
    }

    assert(original_index.size() == 50 && "Original HNSW index should have 50 vectors");

    // Perform query on original index
    std::vector<float> query = {25.0f, 50.0f, 75.0f, 12.5f};
    int k = 5;
    auto orig_results = original_index.search(query, k);
    assert(orig_results.size() == static_cast<size_t>(k) && "Search should return k results");

    // 2. Save HNSW index to file
    bool save_ok = original_index.save(filename);
    assert(save_ok && "HNSW index save should succeed");

    // 3. Load HNSW index into a new instance
    vectordb::HNSWIndex loaded_index;
    bool load_ok = loaded_index.load(filename);
    assert(load_ok && "HNSW index load should succeed");
    assert(loaded_index.size() == 50 && "Loaded HNSW index should have 50 vectors");

    // 4. Perform search on loaded index and compare results
    auto loaded_results = loaded_index.search(query, k);
    assert(loaded_results.size() == orig_results.size() && "Result counts should match");

    for (size_t i = 0; i < orig_results.size(); ++i) {
        assert(orig_results[i].id == loaded_results[i].id && "Result vector IDs should be identical");
        assert(std::fabs(orig_results[i].distance - loaded_results[i].distance) < 1e-5f && "Distances should match");
    }

    // Clean up
    std::remove(filename.c_str());
    std::cout << "[PASS] HNSW Index Persistence Test Passed!" << std::endl;
}

void test_flat_persistence() {
    std::cout << "[TEST] Running Flat Index Persistence Test..." << std::endl;

    const std::string filename = "temp_test_flat.bin";
    std::remove(filename.c_str());

    vectordb::FlatIndex original_flat;
    original_flat.add_vector({101, {1.0f, 2.0f, 3.0f}});
    original_flat.add_vector({102, {4.0f, 5.0f, 6.0f}});

    assert(original_flat.save(filename) && "Flat index save should succeed");

    vectordb::FlatIndex loaded_flat;
    assert(loaded_flat.load(filename) && "Flat index load should succeed");
    assert(loaded_flat.size() == 2 && "Loaded flat index size should be 2");

    auto results = loaded_flat.search({1.1f, 2.1f, 3.1f}, 1);
    assert(!results.empty() && results[0].id == 101 && "Flat index search result should match ID 101");

    std::remove(filename.c_str());
    std::cout << "[PASS] Flat Index Persistence Test Passed!" << std::endl;
}

void test_persistence_error_handling() {
    std::cout << "[TEST] Running Persistence Error Handling Test..." << std::endl;

    vectordb::HNSWIndex index;
    assert(!index.load("non_existent_file_12345.bin") && "Loading non-existent file must fail");

    // Write corrupted file (invalid magic number)
    const std::string bad_filename = "temp_bad_magic.bin";
    std::ofstream out(bad_filename, std::ios::binary);
    uint32_t bad_magic = 0xDEADBEEF;
    out.write(reinterpret_cast<const char*>(&bad_magic), sizeof(bad_magic));
    out.close();

    assert(!index.load(bad_filename) && "Loading corrupted magic file must fail");
    std::remove(bad_filename.c_str());

    std::cout << "[PASS] Persistence Error Handling Test Passed!" << std::endl;
}

int main() {
    test_hnsw_persistence();
    test_flat_persistence();
    test_persistence_error_handling();
    std::cout << "\n[ALL STORAGE TESTS PASSED]" << std::endl;
    return 0;
}
