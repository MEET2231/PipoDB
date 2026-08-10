#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <shared_mutex>
#include "vectordb/Collection.h"

namespace vectordb {

    struct DBQueryRequest {
        std::string collection_name;
        std::vector<float> query_vector;
        int top_k = 10;
        bool include_payload = true;
    };

    struct DBQueryResponse {
        std::vector<CollectionHit> hits;
        double query_time_ms = 0.0;
        bool success = false;
        std::string error_message;
    };

    struct DBUpsertResponse {
        UpsertResult result;
        bool success = false;
        std::string error_message;
    };

    class Database {
    private:
        std::string db_path_;
        std::unordered_map<std::string, std::shared_ptr<Collection>> collections_;
        mutable std::shared_mutex db_mutex_;

        bool save_catalog() const;
        bool load_catalog();

    public:
        explicit Database(const std::string& db_path = "./pipodb_data");
        ~Database();

        // Lifecycle
        bool open();
        bool close();
        bool snapshot() const;

        // DDL (Collection Management)
        bool create_collection(const CollectionParams& params);
        bool drop_collection(const std::string& collection_name);
        bool has_collection(const std::string& collection_name) const;
        std::vector<std::string> list_collections() const;
        std::shared_ptr<Collection> get_collection(const std::string& collection_name) const;

        // DML (Vector Query & Routing)
        VectorID insert_vector(const std::string& collection_name, const std::vector<float>& data, const std::string& payload_json = "", VectorID explicit_id = 0);
        bool insert_vector(const std::string& collection_name, VectorID id, const std::vector<float>& data, const std::string& payload_json);
        bool remove_vector(const std::string& collection_name, VectorID id);

        DBUpsertResponse upsert_if_close(
            const std::string& collection_name,
            const std::vector<float>& data,
            const std::string& payload_json = "",
            float distance_threshold = 0.05f,
            VectorID explicit_id = 0
        );

        DBQueryResponse search(const DBQueryRequest& request) const;
    };

}
