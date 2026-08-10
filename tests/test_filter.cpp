#include <iostream>
#include <cassert>
#include "vectordb/Filter.h"
#include "vectordb/Collection.h"

void test_filter_evaluator() {
    std::cout << "[TEST] Running PayloadFilter Unit Tests..." << std::endl;

    std::string json_sample = "{\"category\": \"tech\", \"year\": 2024, \"author\": \"Bjarne Stroustrup\", \"active\": true}";

    // 1. Test EQ
    vectordb::PayloadFilter filter_eq;
    filter_eq.must.push_back({"category", vectordb::FilterOp::EQ, "tech"});
    assert(filter_eq.matches(json_sample) == true);

    vectordb::PayloadFilter filter_eq_fail;
    filter_eq_fail.must.push_back({"category", vectordb::FilterOp::EQ, "finance"});
    assert(filter_eq_fail.matches(json_sample) == false);

    // 2. Test Numeric GT / GTE / LT / LTE
    vectordb::PayloadFilter filter_gt;
    filter_gt.must.push_back({"year", vectordb::FilterOp::GT, "2020"});
    assert(filter_gt.matches(json_sample) == true);

    vectordb::PayloadFilter filter_lt;
    filter_lt.must.push_back({"year", vectordb::FilterOp::LT, "2020"});
    assert(filter_lt.matches(json_sample) == false);

    // 3. Test CONTAINS
    vectordb::PayloadFilter filter_contains;
    filter_contains.must.push_back({"author", vectordb::FilterOp::CONTAINS, "Bjarne"});
    assert(filter_contains.matches(json_sample) == true);

    // 4. Test SHOULD (OR logic)
    vectordb::PayloadFilter filter_or;
    filter_or.should.push_back({"category", vectordb::FilterOp::EQ, "finance"});
    filter_or.should.push_back({"category", vectordb::FilterOp::EQ, "tech"});
    assert(filter_or.matches(json_sample) == true);

    std::cout << "[PASS] PayloadFilter Unit Tests Passed!" << std::endl;
}

void test_filtered_vector_search() {
    std::cout << "[TEST] Running Filtered Vector Search Test..." << std::endl;

    vectordb::CollectionParams params;
    params.name = "filter_col";
    params.dimension = 3;
    params.index_type = "HNSW";

    vectordb::Collection col(params);

    // Insert 5 vectors: 3 tech (2024), 2 finance (2020)
    col.add_vector(1, {1.0f, 1.0f, 1.0f}, "{\"category\": \"tech\", \"year\": 2024}");
    col.add_vector(2, {1.1f, 1.1f, 1.1f}, "{\"category\": \"finance\", \"year\": 2020}");
    col.add_vector(3, {1.2f, 1.2f, 1.2f}, "{\"category\": \"tech\", \"year\": 2024}");
    col.add_vector(4, {1.3f, 1.3f, 1.3f}, "{\"category\": \"finance\", \"year\": 2020}");
    col.add_vector(5, {1.4f, 1.4f, 1.4f}, "{\"category\": \"tech\", \"year\": 2024}");

    assert(col.size() == 5);

    // Filter: category == "tech" AND year >= 2023
    vectordb::PayloadFilter filter;
    filter.must.push_back({"category", vectordb::FilterOp::EQ, "tech"});
    filter.must.push_back({"year", vectordb::FilterOp::GTE, "2023"});

    auto hits = col.search({1.0f, 1.0f, 1.0f}, 5, true, filter);

    assert(hits.size() == 3);
    for (const auto& hit : hits) {
        assert(hit.payload_json.find("\"category\": \"tech\"") != std::string::npos);
        assert(hit.id == 1 || hit.id == 3 || hit.id == 5);
    }

    std::cout << "[PASS] Filtered Vector Search Test Passed!" << std::endl;
}

int main() {
    test_filter_evaluator();
    test_filtered_vector_search();
    std::cout << "\n[ALL FILTER TESTS PASSED]" << std::endl;
    return 0;
}
