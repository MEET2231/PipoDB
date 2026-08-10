#include "vectordb/SQ8Index.h"
#include <cmath>
#include <algorithm>
#include <queue>
#include <stdexcept>

namespace vectordb {

    SQ8Index::SQ8Index(size_t dimension, MetricType metric)
        : dimension_(dimension), metric_(metric) {}

    QuantizedVector SQ8Index::quantize(VectorID id, const std::vector<float>& float_vec) const {
        float v_min = float_vec[0];
        float v_max = float_vec[0];
        for (size_t d = 1; d < dimension_; ++d) {
            if (float_vec[d] < v_min) v_min = float_vec[d];
            if (float_vec[d] > v_max) v_max = float_vec[d];
        }

        float range = v_max - v_min;
        float v_scale = (range <= 1e-7f) ? (1.0f / 255.0f) : (range / 255.0f);

        std::vector<uint8_t> qdata(dimension_);
        for (size_t d = 0; d < dimension_; ++d) {
            float normalized = (float_vec[d] - v_min) / v_scale;
            qdata[d] = static_cast<uint8_t>(std::clamp(std::round(normalized), 0.0f, 255.0f));
        }

        return QuantizedVector{ id, v_min, v_scale, std::move(qdata) };
    }

    float SQ8Index::compute_asymmetric_distance(const std::vector<float>& query, const QuantizedVector& qvec) const {
        std::vector<float> reconstructed(dimension_);
        for (size_t d = 0; d < dimension_; ++d) {
            reconstructed[d] = qvec.v_min + (static_cast<float>(qvec.data[d]) * qvec.v_scale);
        }
        return Distance::compute(query, reconstructed, metric_);
    }

    void SQ8Index::add_vector(const Vector& vec) {
        if (vec.size() != dimension_) {
            throw std::invalid_argument("Vector dimension does not match SQ8Index dimension.");
        }
        vectors_.push_back(quantize(vec.id, vec.data));
    }

    bool SQ8Index::remove_vector(VectorID id) {
        auto it = std::remove_if(vectors_.begin(), vectors_.end(), [id](const QuantizedVector& qv) {
            return qv.id == id;
        });
        if (it != vectors_.end()) {
            vectors_.erase(it, vectors_.end());
            return true;
        }
        return false;
    }

    size_t SQ8Index::size() const {
        return vectors_.size();
    }

    size_t SQ8Index::memory_bytes() const {
        return vectors_.size() * (sizeof(VectorID) + 2 * sizeof(float) + dimension_ * sizeof(uint8_t));
    }

    std::vector<SearchResult> SQ8Index::search(const std::vector<float>& query, int k) const {
        if (vectors_.empty() || k <= 0) {
            return {};
        }

        if (query.size() != dimension_) {
            throw std::invalid_argument("Query vector dimension does not match SQ8Index dimension.");
        }

        auto cmp = [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        };
        std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp)> max_heap(cmp);

        for (const auto& qvec : vectors_) {
            float dist = compute_asymmetric_distance(query, qvec);
            max_heap.push(SearchResult{ dist, qvec.id });

            if (max_heap.size() > static_cast<size_t>(k)) {
                max_heap.pop();
            }
        }

        std::vector<SearchResult> results;
        results.reserve(max_heap.size());
        while (!max_heap.empty()) {
            results.push_back(max_heap.top());
            max_heap.pop();
        }

        std::reverse(results.begin(), results.end());
        return results;
    }

} // namespace vectordb
