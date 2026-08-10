#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <shared_mutex>
#include <atomic>
#include "vectordb/Vector.h"
#include "vectordb/Index.h"
#include "vectordb/Filter.h"
#include "vectordb/Batch.h"

namespace vectordb {

    struct CollectionParams {
        std::string name;
        size_t dimension = 128;
        std::string metric = "L2";        // "L2", "COSINE"
        std::string index_type = "HNSW";  // "HNSW", "FLAT"
        size_t M = 16;
        size_t ef_construction = 200;
        size_t ef_search = 50;
    };

    struct CollectionHit {
        VectorID id;
        float distance;
        std::string payload_json;
    };

    struct UpsertResult {
        VectorID id;           // Assigned VectorID (either updated or newly inserted)
        bool is_updated;       // true if an existing near vector was updated; false if inserted
        float distance;        // Distance to the nearest neighbor found (-1.0f if collection was empty)
    };

    class Collection {
    private:
        CollectionParams params_;
        std::unique_ptr<Index> index_;
        std::unordered_map<VectorID, std::string> payloads_;
        std::atomic<VectorID> next_auto_id_{1};
        mutable std::shared_mutex collection_mutex_;

    public:
        explicit Collection(const CollectionParams& params);
        ~Collection() = default;

        // Information & Configuration
        const CollectionParams& params() const { return params_; }
        size_t size() const;
        size_t dimension() const { return params_.dimension; }

        // Operations
        // Returns assigned VectorID. If explicit_id == 0, auto-generates ID.
        VectorID add_vector(const std::vector<float>& data, const std::string& payload_json = "", VectorID explicit_id = 0);
        bool add_vector(VectorID id, const std::vector<float>& data, const std::string& payload_json);
        bool remove_vector(VectorID id);

        // High-Throughput Parallel Batch Ingestion
        BatchIngestResult insert_batch(const std::vector<BatchVectorRecord>& records, size_t num_threads = 0);

        // Semantic Near-Duplicate Upsert: Updates existing vector payload if distance <= distance_threshold
        UpsertResult upsert_if_close(
            const std::vector<float>& data,
            const std::string& payload_json = "",
            float distance_threshold = 0.05f,
            VectorID explicit_id = 0
        );

        std::vector<CollectionHit> search(const std::vector<float>& query, int k, bool include_payload = true, const PayloadFilter& filter = {}) const;
        bool get_vector(VectorID id, std::vector<float>& out_data, std::string& out_payload) const;

        // Serialization
        bool save_to_dir(const std::string& dir_path) const;
        bool load_from_dir(const std::string& dir_path);
    };

}
