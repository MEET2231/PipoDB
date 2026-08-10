#include <iostream>
#include <cassert>
#include "vectordb/DB.h"

void test_database_ddl_dml() {
    std::cout << "[TEST] Running Database DDL & DML Test..." << std::endl;

    std::string db_dir = "./temp_db_test";
    vectordb::Database db(db_dir);

    // Create Collections
    vectordb::CollectionParams p1;
    p1.name = "docs";
    p1.dimension = 3;
    p1.index_type = "HNSW";

    vectordb::CollectionParams p2;
    p2.name = "users";
    p2.dimension = 2;
    p2.index_type = "FLAT";

    assert(db.create_collection(p1));
    assert(db.create_collection(p2));
    assert(!db.create_collection(p1)); // Duplicate creation fails

    assert(db.has_collection("docs"));
    assert(db.has_collection("users"));
    assert(db.list_collections().size() == 2);

    // Insert Vectors
    assert(db.insert_vector("docs", 1, {1.0f, 2.0f, 3.0f}, "{\"text\": \"Doc 1\"}"));
    assert(db.insert_vector("docs", 2, {9.0f, 9.0f, 9.0f}, "{\"text\": \"Doc 2\"}"));

    // Search
    vectordb::DBQueryRequest q;
    q.collection_name = "docs";
    q.query_vector = {1.1f, 2.1f, 3.1f};
    q.top_k = 1;
    q.include_payload = true;

    auto res = db.search(q);
    assert(res.success);
    assert(res.hits.size() == 1);
    assert(res.hits[0].id == 1);
    assert(res.hits[0].payload_json == "{\"text\": \"Doc 1\"}");
    assert(res.query_time_ms >= 0.0);

    // Snapshot & Close
    assert(db.snapshot());
    db.close();

    // Reopen Database and test persistence
    vectordb::Database db2(db_dir);
    assert(db2.open());
    assert(db2.has_collection("docs"));
    assert(db2.has_collection("users"));

    auto res2 = db2.search(q);
    assert(res2.success);
    assert(res2.hits.size() == 1);
    assert(res2.hits[0].id == 1);

    // Drop Collection
    assert(db2.drop_collection("users"));
    assert(!db2.has_collection("users"));

    std::cout << "[PASS] Database DDL & DML Test Passed!" << std::endl;
}

int main() {
    test_database_ddl_dml();
    std::cout << "\n[ALL DATABASE ENGINE TESTS PASSED]" << std::endl;
    return 0;
}
