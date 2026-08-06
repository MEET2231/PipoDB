#include <iostream>
#include <vector>
#include "vectordb/Vector.h"
#include "vectordb/FlatIndex.h"

int main() {
    std::cout << "Initializing VectorDB..." << std::endl;

    // Create the index
    vectordb::FlatIndex index;

    // Add some 3D vectors to our database
    index.add_vector({1, {1.0f, 1.0f, 1.0f}});
    index.add_vector({2, {2.0f, 2.0f, 2.0f}});
    index.add_vector({3, {9.0f, 9.0f, 9.0f}});
    index.add_vector({4, {2.1f, 2.1f, 2.1f}});

    std::cout << "Indexed " << index.size() << " vectors." << std::endl;

    // Create a query vector
    std::vector query = {2.0f, 2.0f, 2.0f};
    int k = 2; // We want the top 2 closest matches

    std::cout << "Searching for top " << k << " nearest neighbors..." << std::endl;
    
    // Perform the search
    std::vector results = index.search(query, k);

    // Print the results
    for (const auto& res : results) {
        std::cout << "Vector ID: " << res.id << " | L2 Distance: " << res.distance << std::endl;
    }

    return 0;
}