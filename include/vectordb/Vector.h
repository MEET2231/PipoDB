#pragma once
#include <vector>
#include <cstdint>

namespace vectordb 
{

    using VectorID = std::uint64_t;
    struct Vector
    {   VectorID id;
        std::vector<float> data;
        size_t size() const 
        {
            return data.size();
        }
    };
    struct SearchResult 
    {
        float distance;
        VectorID id;
    };

}
