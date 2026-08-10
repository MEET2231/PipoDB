#include <iostream>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <sys/stat.h>
#include "vectordb/Collection.h"

void test_collection_operations() {
    std::cout << "[TEST] Running Collection Operations Test..." << std::endl;

    vectordb::CollectionParams params;
    params.name = "test_collection";
    params.dimension = 4;
    params.index_type = "HNSW";
    params.M = 16;

    vectordb::Collection collection(params);
    assert(collection.size() == 0);

    collection.add_vector(1, {1.0f, 0.0f, 0.0f, 0.0f}, "{\"title\": \"Vector 1\"}");
    collection.add_vector(2, {0.0f, 1.0f, 0.0f, 0.0f}, "{\"title\": \"Vector 2\"}");
    collection.add_vector(3, {1.1f, 0.1f, 0.0f, 0.0f}, "{\"title\": \"Vector 3\"}");

    assert(collection.size() == 3);

    // Search query
    auto hits = collection.search({1.0f, 0.0f, 0.0f, 0.0f}, 2, true);
    assert(hits.size() == 2);
    assert(hits[0].id == 1);
    assert(hits[0].payload_json == "{\"title\": \"Vector 1\"}");
    assert(hits[1].id == 3);

    std::cout << "[PASS] Collection Operations Test Passed!" << std::endl;
}

void test_auto_id_generation() {
    std::cout << "[TEST] Running Auto-ID Generation & INT_MAX Edge Case Test..." << std::endl;

    vectordb::CollectionParams params;
    params.name = "auto_id_col";
    params.dimension = 2;
    params.index_type = "HNSW";

    vectordb::Collection collection(params);

    // 1. Omit ID: Auto-generates ID 1
    vectordb::VectorID id1 = collection.add_vector({1.0f, 1.0f}, "{\"auto\": 1}");
    assert(id1 == 1);

    // 2. Omit ID: Auto-generates ID 2
    vectordb::VectorID id2 = collection.add_vector({2.0f, 2.0f}, "{\"auto\": 2}");
    assert(id2 == 2);

    // 3. Insert large explicit ID (INT_MAX = 2147483647)
    vectordb::VectorID explicit_id = 2147483647;
    vectordb::VectorID id_large = collection.add_vector({9.0f, 9.0f}, "{\"large\": true}", explicit_id);
    assert(id_large == explicit_id);

    // 4. Omit ID again: Auto-generates ID 3 (does NOT skip to INT_MAX + 1!)
    vectordb::VectorID id3 = collection.add_vector({3.0f, 3.0f}, "{\"auto\": 3}");
    assert(id3 == 3);

    // 5. Try inserting duplicate explicit ID (should throw std::invalid_argument)
    bool exception_caught = false;
    try {
        collection.add_vector({9.0f, 9.0f}, "{\"duplicate\": true}", explicit_id);
    } catch (const std::invalid_argument&) {
        exception_caught = true;
    }
    assert(exception_caught && "Inserting duplicate explicit ID must throw exception");

    assert(collection.size() == 4);

    std::cout << "[PASS] Auto-ID Generation & INT_MAX Edge Case Test Passed!" << std::endl;
}

void test_upsert_if_close() {
    std::cout << "[TEST] Running Semantic Near-Duplicate Upsert Test..." << std::endl;

    vectordb::CollectionParams params;
    params.name = "upsert_col";
    params.dimension = 3;
    params.metric = "L2";
    params.index_type = "HNSW";

    vectordb::Collection collection(params);

    // 1. Initial insert of V1 {1.0, 1.0, 1.0}
    auto u1 = collection.upsert_if_close({1.0f, 1.0f, 1.0f}, "{\"version\": 1}", 0.1f);
    assert(u1.is_updated == false);
    assert(u1.id == 1);
    assert(collection.size() == 1);

    // 2. Insert V2 {1.01, 1.01, 1.01} (Dist ~ 0.0173 <= 0.1) -> Should UPDATE V1
    auto u2 = collection.upsert_if_close({1.01f, 1.01f, 1.01f}, "{\"version\": 2, \"updated\": true}", 0.1f);
    assert(u2.is_updated == true);
    assert(u2.id == 1); // Returns existing ID 1
    assert(collection.size() == 1); // Size remains 1

    // Verify payload updated
    auto hits = collection.search({1.0f, 1.0f, 1.0f}, 1, true);
    assert(hits[0].payload_json == "{\"version\": 2, \"updated\": true}");

    // 3. Insert V3 {5.0, 5.0, 5.0} (Dist > 0.1) -> Should INSERT as new Vector
    auto u3 = collection.upsert_if_close({5.0f, 5.0f, 5.0f}, "{\"version\": 1, \"new\": true}", 0.1f);
    assert(u3.is_updated == false);
    assert(u3.id == 2);
    assert(collection.size() == 2);

    std::cout << "[PASS] Semantic Near-Duplicate Upsert Test Passed!" << std::endl;
}

void test_collection_persistence() {
    std::cout << "[TEST] Running Collection Persistence Test..." << std::endl;

    std::string test_dir = "temp_col_dir";

    vectordb::CollectionParams params;
    params.name = "persist_col";
    params.dimension = 3;
    params.index_type = "HNSW";

    vectordb::Collection original(params);
    original.add_vector(101, {1.0f, 2.0f, 3.0f}, "{\"meta\": \"abc\"}");
    original.add_vector(102, {4.0f, 5.0f, 6.0f}, "{\"meta\": \"xyz\"}");

    assert(original.save_to_dir(test_dir) && "Failed to save collection");

    vectordb::Collection loaded(params);
    assert(loaded.load_from_dir(test_dir) && "Failed to load collection");

    assert(loaded.size() == 2);
    auto hits = loaded.search({1.0f, 2.0f, 3.0f}, 1, true);
    assert(!hits.empty());
    assert(hits[0].id == 101);
    assert(hits[0].payload_json == "{\"meta\": \"abc\"}");

    std::cout << "[PASS] Collection Persistence Test Passed!" << std::endl;
}

int main() {
    test_collection_operations();
    test_auto_id_generation();
    test_upsert_if_close();
    test_collection_persistence();
    std::cout << "\n[ALL COLLECTION TESTS PASSED]" << std::endl;
    return 0;
}
