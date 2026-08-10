#include <iostream>
#include <vector>
#include <cassert>
#include "vectordb/DB.h"

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "        PipoDB Vector Database Engine Demo        " << std::endl;
    std::cout << "==================================================" << std::endl;

    std::string db_path = "./pipodb_demo_data";
    vectordb::Database db(db_path);

    std::cout << "\n1. Creating Collections..." << std::endl;
    vectordb::CollectionParams doc_params;
    doc_params.name = "documents";
    doc_params.dimension = 3;
    doc_params.index_type = "HNSW";
    doc_params.M = 16;

    vectordb::CollectionParams user_params;
    user_params.name = "users";
    user_params.dimension = 3;
    user_params.index_type = "FLAT";

    bool c1 = db.create_collection(doc_params);
    bool c2 = db.create_collection(user_params);
    std::cout << "Collection 'documents' created: " << (c1 ? "YES" : "NO") << std::endl;
    std::cout << "Collection 'users' created: " << (c2 ? "YES" : "NO") << std::endl;

    auto collections = db.list_collections();
    std::cout << "Active Database Collections (" << collections.size() << "): ";
    for (const auto& name : collections) {
        std::cout << name << " ";
    }
    std::cout << std::endl;

    std::cout << "\n2. Inserting Vectors with JSON Payload Metadata..." << std::endl;
    db.insert_vector("documents", 101, {1.0f, 1.0f, 1.0f}, "{\"title\": \"Introduction to C++\", \"author\": \"Bjarne\", \"category\": \"tech\"}");
    db.insert_vector("documents", 102, {2.0f, 2.0f, 2.0f}, "{\"title\": \"Vector Databases 101\", \"author\": \"PipoDB\", \"category\": \"tech\"}");
    db.insert_vector("documents", 103, {9.0f, 9.0f, 9.0f}, "{\"title\": \"Financial Analytics\", \"author\": \"Quant Team\", \"category\": \"finance\"}");

    auto doc_col = db.get_collection("documents");
    std::cout << "Total vectors in 'documents': " << doc_col->size() << std::endl;

    std::cout << "\n3. Performing Similarity Search Query..." << std::endl;
    vectordb::DBQueryRequest req;
    req.collection_name = "documents";
    req.query_vector = {2.1f, 2.1f, 2.1f};
    req.top_k = 2;
    req.include_payload = true;

    auto res = db.search(req);
    std::cout << "Search Status: " << (res.success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Query Execution Time: " << res.query_time_ms << " ms" << std::endl;
    std::cout << "Top " << res.hits.size() << " Unfiltered Search Results:" << std::endl;
    for (const auto& hit : res.hits) {
        std::cout << "  - Vector ID: " << hit.id 
                  << " | Distance: " << hit.distance 
                  << " | Payload: " << hit.payload_json << std::endl;
    }

    std::cout << "\n4. Testing Metadata Payload Filtering (PayloadFilter)..." << std::endl;
    vectordb::DBQueryRequest filtered_req = req;
    filtered_req.filter.must.push_back({"category", vectordb::FilterOp::EQ, "finance"});

    auto filter_res = db.search(filtered_req);
    std::cout << "Filtered Search (category == 'finance') Top Result:" << std::endl;
    std::cout << "  - Vector ID: " << filter_res.hits[0].id 
              << " | Distance: " << filter_res.hits[0].distance 
              << " | Payload: " << filter_res.hits[0].payload_json << std::endl;

    std::cout << "\n5. Testing Semantic Near-Duplicate Upsert (upsert_if_close)..." << std::endl;
    std::cout << "Inserting near-duplicate vector {2.01, 2.01, 2.01} with distance_threshold = 0.1..." << std::endl;
    auto up_res = db.upsert_if_close("documents", {2.01f, 2.01f, 2.01f}, "{\"title\": \"Vector Databases 101 (Revised Edition)\", \"category\": \"tech\"}", 0.1f);
    std::cout << "Upsert Result -> ID: " << up_res.result.id 
              << " | Is Updated: " << (up_res.result.is_updated ? "YES (Merged Near-Duplicate)" : "NO (New Insert)") 
              << " | Distance: " << up_res.result.distance << std::endl;

    std::cout << "\n6. Testing Vector Deletion & Graph Edge Repair..." << std::endl;
    std::cout << "Deleting Vector ID 102 from 'documents'..." << std::endl;
    bool removed = db.remove_vector("documents", 102);
    std::cout << "Vector 102 removed: " << (removed ? "YES" : "NO") << std::endl;
    std::cout << "Remaining vectors in 'documents': " << doc_col->size() << std::endl;

    auto res_after = db.search(req);
    std::cout << "New Top Result Vector ID: " << res_after.hits[0].id 
              << " | Distance: " << res_after.hits[0].distance 
              << " | Payload: " << res_after.hits[0].payload_json << std::endl;

    std::cout << "\n7. Snapshotting Database Catalog & Collections to disk..." << std::endl;
    bool snap = db.snapshot();
    std::cout << "Snapshot saved to " << db_path << ": " << (snap ? "SUCCESS" : "FAILED") << std::endl;

    std::cout << "\n8. Re-opening Database in fresh instance..." << std::endl;
    vectordb::Database db_reopened(db_path);
    db_reopened.open();
    std::cout << "Reopened collections count: " << db_reopened.list_collections().size() << std::endl;

    auto re_res = db_reopened.search(req);
    std::cout << "Reopened Index Top Match Vector ID: " << re_res.hits[0].id 
              << " | Payload: " << re_res.hits[0].payload_json << std::endl;

    std::cout << "\n==================================================" << std::endl;
    std::cout << " [SUCCESS] PipoDB Engine Feature Demo Complete! " << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}