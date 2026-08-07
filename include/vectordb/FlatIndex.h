#pragma once
#include <vector>
#include <string>
#include "vectordb/Index.h"

namespace vectordb {

    class FlatIndex : public Index {
    private:
        std::vector<Vector> vectors_;

    public:
        FlatIndex() = default;
        ~FlatIndex() override = default;

        void add_vector(const Vector& vec) override;
        std::vector<SearchResult> search(const std::vector<float>& query, int k) const override;
        size_t size() const override;
        bool save(const std::string& filepath) const;
        bool load(const std::string& filepath);
    };

} // namespace vectordb