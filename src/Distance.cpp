#include "vectordb/Distance.h"
#include <cmath>
#include <stdexcept>

namespace vectordb {

    float Distance::euclidean(const float* a, const float* b, size_t dim) {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    float Distance::euclidean(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be of the same dimension.");
        }
        return euclidean(a.data(), b.data(), a.size());
    }

} // namespace vectordb