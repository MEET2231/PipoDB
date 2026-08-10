#pragma once

#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <functional>
#include <atomic>
#include "vectordb/Vector.h"

namespace vectordb {

    struct BatchVectorRecord {
        VectorID id = 0; // 0 for auto-generation
        std::vector<float> data;
        std::string payload_json = "";
    };

    struct BatchIngestResult {
        size_t total_count = 0;
        size_t success_count = 0;
        std::vector<VectorID> assigned_ids;
        double elapsed_ms = 0.0;
        double vectors_per_sec = 0.0;
    };

    class IngestWorkerPool {
    private:
        size_t num_threads_;
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;
        std::mutex queue_mutex_;
        std::condition_variable cv_;
        std::condition_variable idle_cv_;
        std::atomic<size_t> pending_tasks_{0};
        std::atomic<bool> stop_{false};

    public:
        explicit IngestWorkerPool(size_t num_threads = 0);
        ~IngestWorkerPool();

        void enqueue(std::function<void()> task);
        void wait_idle();
        size_t num_threads() const { return num_threads_; }
    };

} // namespace vectordb
