#pragma once
#include <vector>
#include "vectordb/Vector.h"

namespace vectordb {

    class Index {
    public:
        virtual ~Index() = default;

        virtual void add_vector(const Vector& vec) = 0;

        virtual std::vector<SearchResult> search(const std::vector<float>& query, int k) const = 0;

        virtual size_t size() const = 0;
    };

}