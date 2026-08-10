#pragma once
#include <cstddef>
#include <vector>
#include <string>

namespace vectordb {

    enum class MetricType {
        L2,
        COSINE,
        IP // Inner Product / Dot Product
    };

    class Distance {
    public:
        static MetricType parse_metric(const std::string& metric_str);

        // Core distance functions taking float pointers
        static float euclidean(const float* a, const float* b, size_t dim);
        static float cosine_distance(const float* a, const float* b, size_t dim);
        static float inner_product_distance(const float* a, const float* b, size_t dim);

        // Vector wrapper overloads
        static float euclidean(const std::vector<float>& a, const std::vector<float>& b);
        static float cosine_distance(const std::vector<float>& a, const std::vector<float>& b);
        static float inner_product_distance(const std::vector<float>& a, const std::vector<float>& b);

        // Generic compute function
        static float compute(const float* a, const float* b, size_t dim, MetricType metric);
        static float compute(const std::vector<float>& a, const std::vector<float>& b, MetricType metric);
    };

} // namespace vectordb