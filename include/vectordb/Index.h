#pragma once
#include <vector>
#include "vectordb/Vector.h"

namespace vectordb {

    class Index {
    public:
        // Virtual destructor ensures proper cleanup of derived classes
        virtual ~Index() = default;

        // Add a new vector to the index
        virtual void add_vector(const Vector& vec) = 0;

        // Search for the top K most similar vectors to the query
        virtual std::vector<SearchResult> search(const std::vector<float>& query, int k) const = 0;
        
        // Return the total number of vectors currently in the index
        virtual size_t size() const = 0;
    };

}