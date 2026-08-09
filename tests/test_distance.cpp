#include <iostream>
#include <cassert>
#include <cmath>
#include "vectordb/Distance.h"

void test_euclidean_distance() {
    std::cout << "[TEST] Running Euclidean Distance Test..." << std::endl;

    std::vector<float> a = {0.0f, 0.0f, 0.0f};
    std::vector<float> b = {3.0f, 4.0f, 0.0f};

    float dist = vectordb::Distance::euclidean(a, b);
    assert(std::fabs(dist - 5.0f) < 1e-5f && "Distance between (0,0,0) and (3,4,0) should be 5.0");

    std::cout << "[PASS] Euclidean Distance Test Passed!" << std::endl;
}

int main() {
    test_euclidean_distance();
    std::cout << "\n[ALL DISTANCE TESTS PASSED]" << std::endl;
    return 0;
}
