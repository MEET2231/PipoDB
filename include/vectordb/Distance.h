#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>

namespace vectordb 
{
    class Distance 
    {
        public:
            static float euclidean(const std::vector<float>& a, const std::vector<float>& b)
            {
                if(a.size() != b.size())
                {
                    throw std::invalid_argument("Vectors must be of the same dimension for Euclidean distance calculation.");
                }
                float sum = 0.0f;
                for(size_t i = 0; i < a.size(); ++i)
                {
                    float diff = a[i] - b[i];
                    sum += diff * diff;
                }
                return std::sqrt(sum);
            }

    };
}
