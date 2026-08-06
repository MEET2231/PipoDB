#pragma once
#include "vectordb/Index.h"
#include "vectordb/Distance.h"
#include <queue>
#include <algorithm>
#include <stdexcept>

namespace vectordb 
{
    class FlatIndex : public Index 
    {
        private:
            std::vector<Vector> vectors_;
        public:
            FlatIndex() = default;
            ~FlatIndex() override = default;

            void add_vector(const Vector& vec) override 
            {
                vectors_.push_back(vec);
            }
            std::vector<SearchResult> search(const std::vector<float>&query,int k) const override
            {
                if(vectors_.empty() || k <= 0)
                {
                    return {};
                }
                size_t dim = query.size();
                if(dim != vectors_[0].size())
                {
                    throw std::invalid_argument("Query vector dimension does not match index dimension.");
                }
                auto cmp = [](const SearchResult& a, const SearchResult& b) {
                    return a.distance < b.distance; 
                };
                std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp)> max_heap(cmp);

                for(size_t i = 0; i < vectors_.size(); ++i)
                {
                    float dist = Distance::euclidean(query, vectors_[i].data);
                    max_heap.push(SearchResult{dist, vectors_[i].id});
                    if(max_heap.size() > static_cast<size_t>(k))
                    {
                        max_heap.pop();
                    }
                }
                std::vector<SearchResult> results;
                while(!max_heap.empty())
                {
                    results.push_back(max_heap.top());
                    max_heap.pop();
                }
                std::reverse(results.begin(), results.end());
                return results;
            }
            size_t size() const override
            {
                return vectors_.size();
            }
    };

};
