#include <iostream>
#include <cassert>
#include <vector>
#include "vectordb/SQ8Index.h"
#include "vectordb/Collection.h"

void test_memory_savings() {
    std::cout << "[TEST] Running SQ8 Memory Savings Test (4x RAM Reduction)..." << std::endl;

    size_t num_vectors = 1000;
    size_t dim = 128;

    vectordb::SQ8Index index(dim);

    for (size_t i = 1; i <= num_vectors; ++i) {
        std::vector<float> vec(dim);
        for (size_t d = 0; d < dim; ++d) {
            vec[d] = static_cast<float>((i + d) % 100) / 10.0f;
        }
        index.add_vector({i, vec});
    }

    assert(index.size() == num_vectors);

    size_t float32_raw_bytes = num_vectors * dim * sizeof(float);
    size_t sq8_ram_bytes = index.memory_bytes();

    double savings_pct = (1.0 - (double)sq8_ram_bytes / (double)float32_raw_bytes) * 100.0;

    std::cout << "  - Vectors Stored     : " << num_vectors << std::endl;
    std::cout << "  - Dimension          : " << dim << std::endl;
    std::cout << "  - Float32 Raw Memory : " << float32_raw_bytes / 1024 << " KB" << std::endl;
    std::cout << "  - SQ8 Quantized RAM  : " << sq8_ram_bytes / 1024 << " KB" << std::endl;
    std::cout << "  - Memory Reduction   : " << savings_pct << "% (4x Savings!)" << std::endl;

    assert(sq8_ram_bytes < float32_raw_bytes / 3 && "SQ8 must achieve at least 3x-4x memory reduction");

    std::cout << "[PASS] SQ8 Memory Savings Test Passed!" << std::endl;
}

void test_sq8_search_recall() {
    std::cout << "[TEST] Running SQ8 Quantized Search Recall Test..." << std::endl;

    vectordb::SQ8Index index(4, vectordb::MetricType::L2);

    index.add_vector({1, {1.0f, 1.0f, 1.0f, 1.0f}});
    index.add_vector({2, {9.0f, 9.0f, 9.0f, 9.0f}});
    index.add_vector({3, {1.05f, 1.05f, 1.05f, 1.05f}});

    assert(index.size() == 3);

    auto hits = index.search({1.0f, 1.0f, 1.0f, 1.0f}, 2);
    assert(hits.size() == 2);
    assert(hits[0].id == 1);
    assert(hits[1].id == 3);

    std::cout << "[PASS] SQ8 Quantized Search Recall Test Passed!" << std::endl;
}

void test_sq8_collection() {
    std::cout << "[TEST] Running SQ8 Collection Integration Test..." << std::endl;

    vectordb::CollectionParams params;
    params.name = "quantized_docs";
    params.dimension = 3;
    params.index_type = "SQ8";
    params.metric = "COSINE";

    vectordb::Collection col(params);
    col.add_vector({1.0f, 0.0f, 0.0f}, "{\"doc\": \"vector 1\"}");
    col.add_vector({0.0f, 1.0f, 0.0f}, "{\"doc\": \"vector 2\"}");

    assert(col.size() == 2);

    auto hits = col.search({0.99f, 0.01f, 0.0f}, 1, true);
    assert(!hits.empty());
    assert(hits[0].id == 1);

    std::cout << "[PASS] SQ8 Collection Integration Test Passed!" << std::endl;
}

int main() {
    test_memory_savings();
    test_sq8_search_recall();
    test_sq8_collection();
    std::cout << "\n[ALL SQ8 TESTS PASSED]" << std::endl;
    return 0;
}
