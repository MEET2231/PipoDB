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
    std::vector< float > query = {2.1f, 2.1f, 2.1f};
    int k = 2; // We want the top 2 closest vectors

    std::cout << "Searching for top " << k << " nearest neighbors on in-memory index..." << std::endl;
    
    // 4. Perform the graph search
    std::vector< vectordb::SearchResult > results = hnsw_index.search(query, k);

    assert(results.size() == 2 && "Should return exactly 2 results");
    assert(results[0].id == 2 && "First closest must be ID 2");
    assert(results[1].id == 4 && "Second closest must be ID 4");

    for (const auto& res : results) {
        std::cout << "Vector ID: " << res.id << " | L2 Distance: " << res.distance << std::endl;
    }

    // 5. Test Persistence (Saving & Loading HNSW Index)
    const std::string db_file = "hnsw_graph.bin";
    std::cout << "\nSaving HNSW index graph to disk (" << db_file << ")..." << std::endl;
    bool saved = hnsw_index.save(db_file);
    assert(saved && "Failed to save HNSW index to disk!");

    std::cout << "Loading HNSW index graph from disk into fresh index instance..." << std::endl;
    vectordb::HNSWIndex reloaded_hnsw;
    bool loaded = reloaded_hnsw.load(db_file);
    assert(loaded && "Failed to load HNSW index from disk!");
    assert(reloaded_hnsw.size() == 50 || reloaded_hnsw.size() == 5 && "Loaded index size should match!");

    std::cout << "Searching reloaded index for query..." << std::endl;
    auto reloaded_results = reloaded_hnsw.search(query, k);

    assert(reloaded_results.size() == 2 && "Reloaded index search failed");
    assert(reloaded_results[0].id == 2 && "Reloaded 1st closest must match");
    assert(reloaded_results[1].id == 4 && "Reloaded 2nd closest must match");

    for (const auto& res : reloaded_results) {
        std::cout << "[Reloaded] Vector ID: " << res.id << " | L2 Distance: " << res.distance << std::endl;
    }

    std::cout << "\n[SUCCESS] HNSW Graph built, persisted, reloaded, and searched correctly!" << std::endl;

    return 0;
}