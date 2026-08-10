#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include "vectordb/Distance.h"

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "    PipoDB SIMD Distance Benchmark Suite         " << std::endl;
    std::cout << "==================================================" << std::endl;

    const size_t num_vectors = 100000;
    const size_t dimension = 128; // Standard embedding dimension

    std::cout << "Generating " << num_vectors << " vectors of dimension " << dimension << "..." << std::endl;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> query(dimension);
    for (size_t i = 0; i < dimension; ++i) query[i] = dist(rng);

    std::vector<std::vector<float>> dataset(num_vectors, std::vector<float>(dimension));
    for (size_t i = 0; i < num_vectors; ++i) {
        for (size_t d = 0; d < dimension; ++d) {
            dataset[i][d] = dist(rng);
        }
    }

    // 1. Benchmark L2 Euclidean Distance
    std::cout << "\nRunning Benchmark: L2 Euclidean Distance..." << std::endl;
    auto start_l2 = std::chrono::high_resolution_clock::now();
    double sum_l2 = 0.0;

    for (size_t i = 0; i < num_vectors; ++i) {
        sum_l2 += vectordb::Distance::euclidean(query.data(), dataset[i].data(), dimension);
    }

    auto end_l2 = std::chrono::high_resolution_clock::now();
    double time_l2_ms = std::chrono::duration<double, std::milli>(end_l2 - start_l2).count();
    double qps_l2 = (num_vectors / time_l2_ms) * 1000.0;

    std::cout << "  - Execution Time : " << std::fixed << std::setprecision(3) << time_l2_ms << " ms" << std::endl;
    std::cout << "  - Calculations/sec: " << std::fixed << std::setprecision(0) << qps_l2 << " ops/sec" << std::endl;

    // 2. Benchmark Cosine Distance
    std::cout << "\nRunning Benchmark: Cosine Distance..." << std::endl;
    auto start_cos = std::chrono::high_resolution_clock::now();
    double sum_cos = 0.0;

    for (size_t i = 0; i < num_vectors; ++i) {
        sum_cos += vectordb::Distance::cosine_distance(query.data(), dataset[i].data(), dimension);
    }

    auto end_cos = std::chrono::high_resolution_clock::now();
    double time_cos_ms = std::chrono::duration<double, std::milli>(end_cos - start_cos).count();
    double qps_cos = (num_vectors / time_cos_ms) * 1000.0;

    std::cout << "  - Execution Time : " << std::fixed << std::setprecision(3) << time_cos_ms << " ms" << std::endl;
    std::cout << "  - Calculations/sec: " << std::fixed << std::setprecision(0) << qps_cos << " ops/sec" << std::endl;

    // 3. Benchmark Inner Product Distance
    std::cout << "\nRunning Benchmark: Inner Product Distance..." << std::endl;
    auto start_ip = std::chrono::high_resolution_clock::now();
    double sum_ip = 0.0;

    for (size_t i = 0; i < num_vectors; ++i) {
        sum_ip += vectordb::Distance::inner_product_distance(query.data(), dataset[i].data(), dimension);
    }

    auto end_ip = std::chrono::high_resolution_clock::now();
    double time_ip_ms = std::chrono::duration<double, std::milli>(end_ip - start_ip).count();
    double qps_ip = (num_vectors / time_ip_ms) * 1000.0;

    std::cout << "  - Execution Time : " << std::fixed << std::setprecision(3) << time_ip_ms << " ms" << std::endl;
    std::cout << "  - Calculations/sec: " << std::fixed << std::setprecision(0) << qps_ip << " ops/sec" << std::endl;

    std::cout << "\n==================================================" << std::endl;
    std::cout << " [SUCCESS] Benchmark Complete! (Checksums: " << sum_l2 + sum_cos + sum_ip << ")" << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}
