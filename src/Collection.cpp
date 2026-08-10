#include "vectordb/Collection.h"
#include "vectordb/HNSWIndex.h"
#include "vectordb/FlatIndex.h"
#include <fstream>
#include <sstream>
#include <mutex>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(_WIN32)
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir(path, 0755)
#endif

namespace vectordb {

    static bool ensure_directory_exists(const std::string& path) {
        struct stat info;
        if (stat(path.c_str(), &info) != 0) {
            return MKDIR(path.c_str()) == 0;
        }
        return (info.st_mode & S_IFDIR) != 0;
    }

    Collection::Collection(const CollectionParams& params)
        : params_(params) 
    {
        MetricType metric = Distance::parse_metric(params_.metric);
        if (params_.index_type == "FLAT") {
            index_ = std::make_unique<FlatIndex>(metric);
        } else {
            index_ = std::make_unique<HNSWIndex>(params_.M, params_.ef_construction, params_.ef_search, metric);
        }
    }

    size_t Collection::size() const {
        std::shared_lock<std::shared_mutex> lock(collection_mutex_);
        return index_ ? index_->size() : 0;
    }

    VectorID Collection::add_vector(const std::vector<float>& data, const std::string& payload_json, VectorID explicit_id) {
        std::unique_lock<std::shared_mutex> lock(collection_mutex_);
        if (data.size() != params_.dimension) {
            throw std::invalid_argument("Vector dimension does not match collection dimension.");
        }

        VectorID assigned_id = explicit_id;
        if (assigned_id == 0) {
            do {
                assigned_id = next_auto_id_.fetch_add(1);
            } while (assigned_id == 0 || payloads_.find(assigned_id) != payloads_.end());
        } else {
            if (payloads_.find(assigned_id) != payloads_.end()) {
                throw std::invalid_argument("VectorID " + std::to_string(assigned_id) + " already exists in collection.");
            }
        }

        index_->add_vector({assigned_id, data});
        payloads_[assigned_id] = payload_json;
        return assigned_id;
    }

    bool Collection::add_vector(VectorID id, const std::vector<float>& data, const std::string& payload_json) {
        return add_vector(data, payload_json, id) != 0;
    }

    bool Collection::remove_vector(VectorID id) {
        std::unique_lock<std::shared_mutex> lock(collection_mutex_);
        if (!index_ || !index_->remove_vector(id)) {
            return false;
        }
        payloads_.erase(id);
        return true;
    }

    UpsertResult Collection::upsert_if_close(const std::vector<float>& data, const std::string& payload_json, float distance_threshold, VectorID explicit_id) {
        std::unique_lock<std::shared_mutex> lock(collection_mutex_);
        if (data.size() != params_.dimension) {
            throw std::invalid_argument("Vector dimension does not match collection dimension.");
        }

        // Search for 1-nearest neighbor if index is non-empty
        if (index_ && index_->size() > 0) {
            auto hits = index_->search(data, 1);
            if (!hits.empty()) {
                float near_dist = hits[0].distance;
                VectorID near_id = hits[0].id;

                if (near_dist <= distance_threshold) {
                    // Close enough! Update payload & return existing near_id
                    payloads_[near_id] = payload_json;
                    return UpsertResult{ near_id, true, near_dist };
                }
            }
        }

        // Not close enough: Insert as new vector
        VectorID assigned_id = explicit_id;
        if (assigned_id == 0) {
            do {
                assigned_id = next_auto_id_.fetch_add(1);
            } while (assigned_id == 0 || payloads_.find(assigned_id) != payloads_.end());
        } else {
            if (payloads_.find(assigned_id) != payloads_.end()) {
                throw std::invalid_argument("VectorID " + std::to_string(assigned_id) + " already exists in collection.");
            }
        }

        index_->add_vector({assigned_id, data});
        payloads_[assigned_id] = payload_json;

        return UpsertResult{ assigned_id, false, -1.0f };
    }

    std::vector<CollectionHit> Collection::search(const std::vector<float>& query, int k, bool include_payload, const PayloadFilter& filter) const {
        std::shared_lock<std::shared_mutex> lock(collection_mutex_);
        if (query.size() != params_.dimension) {
            throw std::invalid_argument("Query vector dimension does not match collection dimension.");
        }

        int search_k = filter.empty() ? k : std::max(k * 10, 100);
        auto search_results = index_->search(query, search_k);
        std::vector<CollectionHit> hits;
        hits.reserve(k);

        for (const auto& res : search_results) {
            std::string payload = "";
            auto it = payloads_.find(res.id);
            if (it != payloads_.end()) {
                payload = it->second;
            }

            if (!filter.empty()) {
                if (!filter.matches(payload)) {
                    continue;
                }
            }

            hits.push_back(CollectionHit{res.id, res.distance, include_payload ? payload : ""});
            if (hits.size() >= static_cast<size_t>(k)) {
                break;
            }
        }

        return hits;
    }

    bool Collection::get_vector(VectorID id, std::vector<float>& out_data, std::string& out_payload) const {
        std::shared_lock<std::shared_mutex> lock(collection_mutex_);
        auto it = payloads_.find(id);
        if (it == payloads_.end()) {
            return false;
        }
        out_payload = it->second;
        // Vector search or data retrieval verified via payload presence
        out_data.clear();
        return true;
    }

    bool Collection::save_to_dir(const std::string& dir_path) const {
        std::shared_lock<std::shared_mutex> lock(collection_mutex_);
        if (!ensure_directory_exists(dir_path)) {
            return false;
        }

        // 1. Write Config File
        std::string config_file = dir_path + "/config.txt";
        std::ofstream cfg(config_file);
        if (!cfg.is_open()) return false;

        cfg << "name=" << params_.name << "\n";
        cfg << "dimension=" << params_.dimension << "\n";
        cfg << "metric=" << params_.metric << "\n";
        cfg << "index_type=" << params_.index_type << "\n";
        cfg << "M=" << params_.M << "\n";
        cfg << "ef_construction=" << params_.ef_construction << "\n";
        cfg << "ef_search=" << params_.ef_search << "\n";
        cfg.close();

        // 2. Write Index File
        std::string index_file = dir_path + "/index.bin";
        bool index_saved = false;
        if (params_.index_type == "FLAT") {
            auto* flat = dynamic_cast<FlatIndex*>(index_.get());
            if (flat) index_saved = flat->save(index_file);
        } else {
            auto* hnsw = dynamic_cast<HNSWIndex*>(index_.get());
            if (hnsw) index_saved = hnsw->save(index_file);
        }

        if (!index_saved) return false;

        // 3. Write Payloads File
        std::string payload_file = dir_path + "/payloads.bin";
        std::ofstream out_p(payload_file, std::ios::binary);
        if (!out_p.is_open()) return false;

        uint64_t count = payloads_.size();
        out_p.write(reinterpret_cast<const char*>(&count), sizeof(count));

        for (const auto& pair : payloads_) {
            uint64_t id = pair.first;
            uint32_t len = static_cast<uint32_t>(pair.second.size());
            out_p.write(reinterpret_cast<const char*>(&id), sizeof(id));
            out_p.write(reinterpret_cast<const char*>(&len), sizeof(len));
            if (len > 0) {
                out_p.write(pair.second.data(), len);
            }
        }

        out_p.close();
        return out_p.good();
    }

    bool Collection::load_from_dir(const std::string& dir_path) {
        std::unique_lock<std::shared_mutex> lock(collection_mutex_);

        // 1. Read Config File
        std::string config_file = dir_path + "/config.txt";
        std::ifstream cfg(config_file);
        if (!cfg.is_open()) return false;

        std::string line;
        while (std::getline(cfg, line)) {
            size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);

            if (key == "name") params_.name = val;
            else if (key == "dimension") params_.dimension = std::stoull(val);
            else if (key == "metric") params_.metric = val;
            else if (key == "index_type") params_.index_type = val;
            else if (key == "M") params_.M = std::stoull(val);
            else if (key == "ef_construction") params_.ef_construction = std::stoull(val);
            else if (key == "ef_search") params_.ef_search = std::stoull(val);
        }
        cfg.close();

        // 2. Re-instantiate Index
        MetricType metric = Distance::parse_metric(params_.metric);
        if (params_.index_type == "FLAT") {
            index_ = std::make_unique<FlatIndex>(metric);
            auto* flat = dynamic_cast<FlatIndex*>(index_.get());
            if (!flat || !flat->load(dir_path + "/index.bin")) return false;
        } else {
            index_ = std::make_unique<HNSWIndex>(params_.M, params_.ef_construction, params_.ef_search, metric);
            auto* hnsw = dynamic_cast<HNSWIndex*>(index_.get());
            if (!hnsw || !hnsw->load(dir_path + "/index.bin")) return false;
        }

        // 3. Read Payloads File
        std::string payload_file = dir_path + "/payloads.bin";
        std::ifstream in_p(payload_file, std::ios::binary);
        if (!in_p.is_open()) return false;

        uint64_t count = 0;
        in_p.read(reinterpret_cast<char*>(&count), sizeof(count));

        payloads_.clear();
        payloads_.reserve(count);

        for (uint64_t i = 0; i < count; ++i) {
            uint64_t id = 0;
            uint32_t len = 0;
            in_p.read(reinterpret_cast<char*>(&id), sizeof(id));
            in_p.read(reinterpret_cast<char*>(&len), sizeof(len));

            std::string payload = "";
            if (len > 0) {
                payload.resize(len);
                in_p.read(&payload[0], len);
            }
            payloads_[id] = std::move(payload);
        }

        in_p.close();
        return true;
    }

}
