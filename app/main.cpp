#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "vectordb/Vector.h"
#include "vectordb/FlatIndex.h"

int main() {
    std::string db_file = "/home/coco/pipo_vectors.vdb";

    std::cout << "--- Step 1: Creating and Saving Index ---" << std::endl;
    vectordb::FlatIndex index;
    index.add_vector({101, {1.0f, 2.0f, 3.0f}});
    index.add_vector({102, {4.0f, 5.0f, 6.0f}});
    
    bool saved = index.save(db_file);
    assert(saved);
    std::cout << "Saved " << index.size() << " vectors to " << db_file << std::endl;

    std::cout << "\n--- Step 2: Loading Index from Disk ---" << std::endl;
    vectordb::FlatIndex loaded_index;
    
    bool loaded = loaded_index.load(db_file);
    assert(loaded);
    std::cout << "Loaded " << loaded_index.size() << " vectors from " << db_file << std::endl;

    std::vector< float > query = {1.0f, 2.0f, 3.0f};
    std::vector< vectordb::SearchResult > results = loaded_index.search(query, 1);
    
    assert(results.size() == 1);
    assert(results[0].id == 101);
    
    std::cout << "[SUCCESS] Reloaded Vector ID: " << results[0].id << std::endl;

    return 0;
}