#pragma once
#include <cstddef>
#include <vector>

namespace vectordb {

    class Distance {
    public:
        static float euclidean(const float* a, const float* b, size_t dim);
        
        static float euclidean(const std::vector<float>& a, const std::vector<float>& b);
    };

} // namespace vectordb