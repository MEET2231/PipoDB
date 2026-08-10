#include "vectordb/DB.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <mutex>
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

    Database::Database(const std::string& db_path)
        : db_path_(db_path) 
    {
        ensure_directory_exists(db_path_);
        ensure_directory_exists(db_path_ + "/collections");
    }

    Database::~Database() {
        close();
    }

    bool Database::save_catalog() const {
        std::string catalog_path = db_path_ + "/catalog.txt";
        std::ofstream out(catalog_path);
        if (!out.is_open()) return false;

        out << collections_.size() << "\n";
        for (const auto& pair : collections_) {
            const auto& p = pair.second->params();
            out << p.name << " "
                << p.dimension << " "
                << p.metric << " "
                << p.index_type << " "
                << p.M << " "
                << p.ef_construction << " "
                << p.ef_search << "\n";
        }

        out.close();
        return out.good();
    }

    bool Database::load_catalog() {
        std::string catalog_path = db_path_ + "/catalog.txt";
        std::ifstream in(catalog_path);
        if (!in.is_open()) return false;

        size_t count = 0;
        if (!(in >> count)) return false;

        collections_.clear();
        for (size_t i = 0; i < count; ++i) {
            CollectionParams p;
            if (!(in >> p.name >> p.dimension >> p.metric >> p.index_type >> p.M >> p.ef_construction >> p.ef_search)) {
                break;
            }

            auto col = std::make_shared<Collection>(p);
            std::string col_dir = db_path_ + "/collections/" + p.name;
            col->load_from_dir(col_dir);
            collections_[p.name] = col;
        }

        in.close();
        return true;
    }

    bool Database::open() {
        std::unique_lock<std::shared_mutex> lock(db_mutex_);
        ensure_directory_exists(db_path_);
        ensure_directory_exists(db_path_ + "/collections");
        return load_catalog();
    }

    bool Database::close() {
        return snapshot();
    }

    bool Database::snapshot() const {
        std::shared_lock<std::shared_mutex> lock(db_mutex_);
        ensure_directory_exists(db_path_);
        ensure_directory_exists(db_path_ + "/collections");

        for (const auto& pair : collections_) {
            std::string col_dir = db_path_ + "/collections/" + pair.first;
            pair.second->save_to_dir(col_dir);
        }

        return save_catalog();
    }

    bool Database::create_collection(const CollectionParams& params) {
        std::unique_lock<std::shared_mutex> lock(db_mutex_);
        if (collections_.find(params.name) != collections_.end()) {
            return false; // Collection already exists
        }

        auto col = std::make_shared<Collection>(params);
        collections_[params.name] = col;

        std::string col_dir = db_path_ + "/collections/" + params.name;
        col->save_to_dir(col_dir);
        save_catalog();
        return true;
    }

    bool Database::drop_collection(const std::string& collection_name) {
        std::unique_lock<std::shared_mutex> lock(db_mutex_);
        auto it = collections_.find(collection_name);
        if (it == collections_.end()) {
            return false;
        }

        collections_.erase(it);
        save_catalog();
        return true;
    }

    bool Database::has_collection(const std::string& collection_name) const {
        std::shared_lock<std::shared_mutex> lock(db_mutex_);
        return collections_.find(collection_name) != collections_.end();
    }

    std::vector<std::string> Database::list_collections() const {
        std::shared_lock<std::shared_mutex> lock(db_mutex_);
        std::vector<std::string> list;
        list.reserve(collections_.size());
        for (const auto& pair : collections_) {
            list.push_back(pair.first);
        }
        return list;
    }

    std::shared_ptr<Collection> Database::get_collection(const std::string& collection_name) const {
        std::shared_lock<std::shared_mutex> lock(db_mutex_);
        auto it = collections_.find(collection_name);
        if (it != collections_.end()) {
            return it->second;
        }
        return nullptr;
    }

    bool Database::insert_vector(const std::string& collection_name, VectorID id, const std::vector<float>& data, const std::string& payload_json) {
        std::shared_ptr<Collection> col;
        {
            std::shared_lock<std::shared_mutex> lock(db_mutex_);
            auto it = collections_.find(collection_name);
            if (it == collections_.end()) {
                return false;
            }
            col = it->second;
        }

        return col->add_vector(id, data, payload_json);
    }

    DBQueryResponse Database::search(const DBQueryRequest& request) const {
        auto start_time = std::chrono::high_resolution_clock::now();

        std::shared_ptr<Collection> col;
        {
            std::shared_lock<std::shared_mutex> lock(db_mutex_);
            auto it = collections_.find(request.collection_name);
            if (it == collections_.end()) {
                return DBQueryResponse{ {}, 0.0, false, "Collection '" + request.collection_name + "' not found." };
            }
            col = it->second;
        }

        try {
            auto hits = col->search(request.query_vector, request.top_k, request.include_payload);
            auto end_time = std::chrono::high_resolution_clock::now();
            double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

            return DBQueryResponse{ hits, duration_ms, true, "" };
        } catch (const std::exception& ex) {
            return DBQueryResponse{ {}, 0.0, false, ex.what() };
        }
    }

}
