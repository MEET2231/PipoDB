#pragma once
#include <vector>
#include <unordered_map>
#include <random>
#include <string>
#include "vectordb/Index.h"
#include "vectordb/Distance.h"

namespace vectordb {

    class Storage;

    class HNSWIndex : public Index {
        friend class Storage;

    private:
        // A single vector and its graph connections
        struct Node {
            VectorID id;
            std::vector< float > data;
            
            // neighbors[level] contains a list of connected VectorIDs at that level
            std::vector< std::vector< VectorID > > neighbors; 
            
            int max_level() const {
                return static_cast< int >(neighbors.size()) - 1;
            }
        };

        // HNSW Hyperparameters
        size_t M_;                 // Max connections per node per layer (e.g., 16)
        size_t M_max_0_;           // Max connections for the bottom layer 0 (usually 2 * M_)
        size_t ef_construction_;   // Candidate list size during insertion (e.g., 200)
        size_t ef_search_;         // Candidate list size during search (e.g., 50)
        double mult_;              // Multiplier for random level generation
        MetricType metric_;        // Distance Metric (L2, COSINE, IP)

        // The Graph Database
        std::unordered_map< VectorID, Node > nodes_;
        
        // Graph state
        VectorID entry_point_id_;
        int max_graph_level_;
        bool is_empty_;

        // Random Number Generator for assigning node levels
        mutable std::mt19937 rng_;
        mutable std::uniform_real_distribution< double > level_dist_;

        // Internal math & logic helpers
        int get_random_level() const;
        std::vector< SearchResult > search_layer(const std::vector< float >& query, VectorID entry_point, int ef, int layer) const;

    public:
        // Constructor sets up the hyperparameters
        HNSWIndex(size_t M = 16, size_t ef_construction = 200, size_t ef_search = 50, MetricType metric = MetricType::L2);
        ~HNSWIndex() override = default;

        MetricType metric() const { return metric_; }

        void add_vector(const Vector& vec) override;
        std::vector< SearchResult > search(const std::vector< float >& query, int k) const override;
        size_t size() const override;

        bool save(const std::string& filepath) const;
        bool load(const std::string& filepath);
    };

}