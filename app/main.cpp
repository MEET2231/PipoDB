#include <iostream>
#include <vector>
#include <cassert>
#include "vectordb/Vector.h"
#include "vectordb/HNSWIndex.h"

int main() {
    std::cout << "--- Initializing HNSW Index Test ---" << std::endl;

    // 1. Create the HNSW index
    // Parameters: M = 16 (max connections), ef_construction = 200, ef_search = 50
    vectordb::HNSWIndex hnsw_index(16, 200, 50);

    // 2. Add test vectors to build the graph
    std::cout << "Adding vectors to the graph..." << std::endl;
    hnsw_index.add_vector({1, {1.0f, 1.0f, 1.0f}});
    hnsw_index.add_vector({2, {2.0f, 2.0f, 2.0f}});
    hnsw_index.add_vector({3, {9.0f, 9.0f, 9.0f}});
    hnsw_index.add_vector({4, {2.5f, 2.5f, 2.5f}});
    hnsw_index.add_vector({5, {0.0f, 0.0f, 0.0f}});

    std::cout << "Total vectors indexed: " << hnsw_index.size() << std::endl;

    // 3. Define a query vector
    // We are looking for neighbors close to {2.1, 2.1, 2.1}
    std::vector< float > query = {2.1f, 2.1f, 2.1f};
    int k = 2; // We want the top 2 closest vectors

    std::cout << "Searching for top " << k << " nearest neighbors..." << std::endl;
    
    // 4. Perform the graph search
    std::vector< vectordb::SearchResult > results = hnsw_index.search(query, k);

    // 5. Assert and Print results
    assert(results.size() == 2 && "Should return exactly 2 results");
    
    // Mathematically, the closest to {2.1, 2.1, 2.1} are:
    // 1st: ID 2 {2.0, 2.0, 2.0} -> Distance ~ 0.1732
    // 2nd: ID 4 {2.5, 2.5, 2.5} -> Distance ~ 0.6928
    assert(results[0].id == 2 && "First closest must be ID 2");
    assert(results[1].id == 4 && "Second closest must be ID 4");

    for (const auto& res : results) {
        std::cout << "Vector ID: " << res.id << " | L2 Distance: " << res.distance << std::endl;
    }

    std::cout << "\n[SUCCESS] HNSW Graph built and searched correctly!" << std::endl;

    return 0;
}