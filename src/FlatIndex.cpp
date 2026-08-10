#include "vectordb/FlatIndex.h"
#include "vectordb/Distance.h"
#include "vectordb/Storage.h"
#include <queue>
#include <algorithm>
#include <stdexcept>

namespace vectordb {

    void FlatIndex::add_vector(const Vector& vec) {
        vectors_.push_back(vec);
    }

    bool FlatIndex::remove_vector(VectorID id) {
        auto it = std::remove_if(vectors_.begin(), vectors_.end(), [id](const Vector& vec) {
            return vec.id == id;
        });
        if (it != vectors_.end()) {
            vectors_.erase(it, vectors_.end());
            return true;
        }
        return false;
    }

    size_t FlatIndex::size() const {
        return vectors_.size();
    }

    std::vector<SearchResult> FlatIndex::search(const std::vector<float>& query, int k) const {
        if (vectors_.empty() || k <= 0) {
            return {};
        }

        size_t dim = query.size();

        if (dim != vectors_[0].size()) {
            throw std::invalid_argument("Query vector dimension does not match index dimension.");
        }

        auto cmp = [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance; 
        };
        std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp)> max_heap(cmp);

        for (const auto& vec : vectors_) {
            float dist = Distance::compute(query, vec.data, metric_);
            max_heap.push(SearchResult{dist, vec.id});

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
    
    bool FlatIndex::save(const std::string& filepath) const {
        return Storage::save_to_file(filepath, vectors_);
    }

    bool FlatIndex::load(const std::string& filepath) {
        return Storage::load_from_file(filepath, vectors_);
    }

} // namespace vectordb