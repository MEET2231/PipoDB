#pragma once
#include <vector>
#include <string>
#include "vectordb/Index.h"
#include "vectordb/Distance.h"

namespace vectordb {

    class FlatIndex : public Index {
    private:
        std::vector<Vector> vectors_;
        MetricType metric_;

    public:
        explicit FlatIndex(MetricType metric = MetricType::L2) : metric_(metric) {}
        ~FlatIndex() override = default;

        MetricType metric() const { return metric_; }

        void add_vector(const Vector& vec) override;
        bool remove_vector(VectorID id) override;
        std::vector<SearchResult> search(const std::vector<float>& query, int k) const override;
        size_t size() const override;
        bool save(const std::string& filepath) const;
        bool load(const std::string& filepath);
    };

} // namespace vectordb