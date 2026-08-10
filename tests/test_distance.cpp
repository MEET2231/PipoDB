#include <iostream>
#include <cassert>
#include <cmath>
#include "vectordb/Distance.h"

void test_euclidean_distance() {
    std::cout << "[TEST] Running Euclidean Distance SIMD Test..." << std::endl;

    // Small vector test
    std::vector<float> a = {0.0f, 0.0f, 0.0f};
    std::vector<float> b = {3.0f, 4.0f, 0.0f};
    float dist = vectordb::Distance::euclidean(a, b);
    assert(std::fabs(dist - 5.0f) < 1e-5f);

    // Large 16-dimensional vector (exercises SIMD 8-float lanes + remainder)
    std::vector<float> v1(16, 1.0f);
    std::vector<float> v2(16, 3.0f); // difference per element = 2.0, squared sum = 16 * 4 = 64, sqrt = 8
    float dist_simd = vectordb::Distance::euclidean(v1, v2);
    assert(std::fabs(dist_simd - 8.0f) < 1e-5f);

    std::cout << "[PASS] Euclidean Distance SIMD Test Passed!" << std::endl;
}

void test_cosine_distance() {
    std::cout << "[TEST] Running Cosine Distance SIMD Test..." << std::endl;

    // Identical vectors (Cosine Distance = 0)
    std::vector<float> v1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> v2 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float dist_same = vectordb::Distance::cosine_distance(v1, v2);
    assert(std::fabs(dist_same - 0.0f) < 1e-5f);

    // Orthogonal vectors (Cosine Distance = 1.0)
    std::vector<float> o1 = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> o2 = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float dist_ortho = vectordb::Distance::cosine_distance(o1, o2);
    assert(std::fabs(dist_ortho - 1.0f) < 1e-5f);

    // Opposite vectors (Cosine Distance = 2.0)
    std::vector<float> op1 = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> op2 = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    float dist_opp = vectordb::Distance::cosine_distance(op1, op2);
    assert(std::fabs(dist_opp - 2.0f) < 1e-5f);

    std::cout << "[PASS] Cosine Distance SIMD Test Passed!" << std::endl;
}

void test_inner_product_distance() {
    std::cout << "[TEST] Running Inner Product Distance Test..." << std::endl;

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> b = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    // Dot product sum(1..9) = 45.0, Distance = -45.0
    float dist_ip = vectordb::Distance::inner_product_distance(a, b);
    assert(std::fabs(dist_ip - (-45.0f)) < 1e-5f);

    std::cout << "[PASS] Inner Product Distance Test Passed!" << std::endl;
}

int main() {
    test_euclidean_distance();
    test_cosine_distance();
    test_inner_product_distance();
    std::cout << "\n[ALL DISTANCE TESTS PASSED]" << std::endl;
    return 0;
}
