#include "vectordb/Batch.h"

namespace vectordb {

    IngestWorkerPool::IngestWorkerPool(size_t num_threads) {
        if (num_threads == 0) {
            num_threads_ = std::thread::hardware_concurrency();
            if (num_threads_ == 0) num_threads_ = 4;
        } else {
            num_threads_ = num_threads;
        }

        for (size_t i = 0; i < num_threads_; ++i) {
            workers_.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->cv_.wait(lock, [this]() {
                            return this->stop_ || !this->tasks_.empty();
                        });

                        if (this->stop_ && this->tasks_.empty()) {
                            return;
                        }

                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    IngestWorkerPool::~IngestWorkerPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void IngestWorkerPool::enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) return;
            pending_tasks_++;
            tasks_.push([this, task]() {
                task();
                pending_tasks_--;
                idle_cv_.notify_all();
            });
        }
        cv_.notify_one();
    }

    void IngestWorkerPool::wait_idle() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        idle_cv_.wait(lock, [this]() {
            return pending_tasks_ == 0 && tasks_.empty();
        });
    }

} // namespace vectordb
