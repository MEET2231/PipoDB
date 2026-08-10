#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include "vectordb/Index.h"
#include "vectordb/Distance.h"

namespace vectordb {

    struct QuantizedVector {
        VectorID id;
        float v_min;
        float v_scale;
        std::vector<uint8_t> data; // 1 byte per dimension (4x RAM reduction!)
    };

    class SQ8Index : public Index {
    private:
        size_t dimension_;
        MetricType metric_;
        std::vector<QuantizedVector> vectors_;

        QuantizedVector quantize(VectorID id, const std::vector<float>& float_vec) const;
        float compute_asymmetric_distance(const std::vector<float>& query, const QuantizedVector& qvec) const;

    public:
        explicit SQ8Index(size_t dimension, MetricType metric = MetricType::L2);
        ~SQ8Index() override = default;

        MetricType metric() const { return metric_; }

        void add_vector(const Vector& vec) override;
        bool remove_vector(VectorID id) override;
        std::vector<SearchResult> search(const std::vector<float>& query, int k) const override;
        size_t size() const override;

        // Returns total RAM memory used by stored vectors in bytes
        size_t memory_bytes() const;
    };

} // namespace vectordb
