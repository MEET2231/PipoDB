#include <iostream>
#include <cassert>
#include <vector>
#include "vectordb/Collection.h"
#include "vectordb/Batch.h"

void test_parallel_batch_ingestion() {
    std::cout << "[TEST] Running High-Throughput Parallel Batch Ingestion Test..." << std::endl;

    vectordb::CollectionParams params;
    params.name = "batch_col";
    params.dimension = 16;
    params.index_type = "HNSW";

    vectordb::Collection col(params);

    size_t num_vectors = 5000;
    std::vector<vectordb::BatchVectorRecord> records;
    records.reserve(num_vectors);

    for (size_t i = 1; i <= num_vectors; ++i) {
        std::vector<float> vec(16);
        for (size_t d = 0; d < 16; ++d) {
            vec[d] = static_cast<float>((i + d) % 50) / 5.0f;
        }
        records.push_back(vectordb::BatchVectorRecord{ 0, vec, "{\"batch_id\":" + std::to_string(i) + "}" });
    }

    std::cout << "  - Ingesting " << num_vectors << " vectors in parallel across thread pool..." << std::endl;
    auto res = col.insert_batch(records);

    std::cout << "  - Total Vectors Requested: " << res.total_count << std::endl;
    std::cout << "  - Successfully Ingested : " << res.success_count << std::endl;
    std::cout << "  - Elapsed Time           : " << res.elapsed_ms << " ms" << std::endl;
    std::cout << "  - Ingestion Throughput   : " << res.vectors_per_sec << " vectors/sec" << std::endl;

    assert(res.success_count == num_vectors);
    assert(col.size() == num_vectors);

    // Perform search query
    std::vector<float> query(16, 1.0f);
    auto hits = col.search(query, 3);
    assert(hits.size() == 3);

    std::cout << "[PASS] High-Throughput Parallel Batch Ingestion Test Passed!" << std::endl;
}

int main() {
    test_parallel_batch_ingestion();
    std::cout << "\n[ALL BATCH INGESTION TESTS PASSED]" << std::endl;
    return 0;
}
