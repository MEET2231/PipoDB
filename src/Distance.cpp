#include "vectordb/Distance.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cctype>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace vectordb {

    MetricType Distance::parse_metric(const std::string& metric_str) {
        std::string upper = metric_str;
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
        if (upper == "COSINE") return MetricType::COSINE;
        if (upper == "IP" || upper == "DOT" || upper == "INNER_PRODUCT") return MetricType::IP;
        return MetricType::L2;
    }

    float Distance::euclidean(const float* a, const float* b, size_t dim) {
        float sum = 0.0f;
        size_t i = 0;

#if defined(__AVX2__)
        __m256 sum_vec = _mm256_setzero_ps();
        for (; i + 7 < dim; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
        }
        alignas(32) float buf[8];
        _mm256_storeu_ps(buf, sum_vec);
        sum = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
#endif

        for (; i < dim; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }

        return std::sqrt(sum);
    }

    float Distance::inner_product_distance(const float* a, const float* b, size_t dim) {
        float dot = 0.0f;
        size_t i = 0;

#if defined(__AVX2__)
        __m256 dot_vec = _mm256_setzero_ps();
        for (; i + 7 < dim; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            dot_vec = _mm256_fmadd_ps(va, vb, dot_vec);
        }
        alignas(32) float buf[8];
        _mm256_storeu_ps(buf, dot_vec);
        dot = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
#endif

        for (; i < dim; ++i) {
            dot += a[i] * b[i];
        }

        return -dot;
    }

    float Distance::cosine_distance(const float* a, const float* b, size_t dim) {
        float dot = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        size_t i = 0;

#if defined(__AVX2__)
        __m256 dot_vec = _mm256_setzero_ps();
        __m256 na_vec = _mm256_setzero_ps();
        __m256 nb_vec = _mm256_setzero_ps();

        for (; i + 7 < dim; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            dot_vec = _mm256_fmadd_ps(va, vb, dot_vec);
            na_vec = _mm256_fmadd_ps(va, va, na_vec);
            nb_vec = _mm256_fmadd_ps(vb, vb, nb_vec);
        }

        alignas(32) float buf_dot[8], buf_na[8], buf_nb[8];
        _mm256_storeu_ps(buf_dot, dot_vec);
        _mm256_storeu_ps(buf_na, na_vec);
        _mm256_storeu_ps(buf_nb, nb_vec);

        for (int k = 0; k < 8; ++k) {
            dot += buf_dot[k];
            norm_a += buf_na[k];
            norm_b += buf_nb[k];
        }
#endif

        for (; i < dim; ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if (norm_a <= 0.0f || norm_b <= 0.0f) {
            return 1.0f;
        }

        float sim = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        if (sim > 1.0f) sim = 1.0f;
        if (sim < -1.0f) sim = -1.0f;

        return 1.0f - sim;
    }

    float Distance::euclidean(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) throw std::invalid_argument("Vectors must be of the same dimension.");
        return euclidean(a.data(), b.data(), a.size());
    }

    float Distance::cosine_distance(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) throw std::invalid_argument("Vectors must be of the same dimension.");
        return cosine_distance(a.data(), b.data(), a.size());
    }

    float Distance::inner_product_distance(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) throw std::invalid_argument("Vectors must be of the same dimension.");
        return inner_product_distance(a.data(), b.data(), a.size());
    }

    float Distance::compute(const float* a, const float* b, size_t dim, MetricType metric) {
        switch (metric) {
            case MetricType::COSINE:
                return cosine_distance(a, b, dim);
            case MetricType::IP:
                return inner_product_distance(a, b, dim);
            case MetricType::L2:
            default:
                return euclidean(a, b, dim);
        }
    }

    float Distance::compute(const std::vector<float>& a, const std::vector<float>& b, MetricType metric) {
        if (a.size() != b.size()) throw std::invalid_argument("Vectors must be of the same dimension.");
        return compute(a.data(), b.data(), a.size(), metric);
    }

} // namespace vectordb