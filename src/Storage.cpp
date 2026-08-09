#include "vectordb/Storage.h"
#include "vectordb/HNSWIndex.h"
#include <fstream>
#include <iostream>

namespace vectordb {

    bool Storage::save_to_file(const std::string& filepath, const std::vector< Vector >& vectors) {
        std::ofstream out(filepath, std::ios::binary);
        if (!out.is_open()) return false;

        uint32_t magic = MAGIC_NUMBER;
        uint64_t count = vectors.size();
        uint32_t dim = count > 0 ? static_cast< uint32_t >(vectors[0].data.size()) : 0;

        out.write(reinterpret_cast< const char* >(&magic), sizeof(magic));
        out.write(reinterpret_cast< const char* >(&count), sizeof(count));
        out.write(reinterpret_cast< const char* >(&dim), sizeof(dim));

        for (const auto& vec : vectors) {
            out.write(reinterpret_cast< const char* >(&vec.id), sizeof(vec.id));
            out.write(reinterpret_cast< const char* >(vec.data.data()), dim * sizeof(float));
        }

        out.close();
        return true;
    }

    bool Storage::load_from_file(const std::string& filepath, std::vector< Vector >& out_vectors) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in.is_open()) return false;

        uint32_t magic = 0;
        uint64_t count = 0;
        uint32_t dim = 0;

        in.read(reinterpret_cast< char* >(&magic), sizeof(magic));
        if (magic != MAGIC_NUMBER) return false;

        in.read(reinterpret_cast< char* >(&count), sizeof(count));
        in.read(reinterpret_cast< char* >(&dim), sizeof(dim));

        out_vectors.clear();
        out_vectors.reserve(count);

        for (uint64_t i = 0; i < count; ++i) {
            Vector vec;
            in.read(reinterpret_cast< char* >(&vec.id), sizeof(vec.id));
            vec.data.resize(dim);
            in.read(reinterpret_cast< char* >(vec.data.data()), dim * sizeof(float));
            out_vectors.push_back(std::move(vec));
        }

        in.close();
        return true;
    }

    bool Storage::save_hnsw_to_file(const std::string& filepath, const HNSWIndex& index) {
        std::ofstream out(filepath, std::ios::binary);
        if (!out.is_open()) return false;

        uint32_t magic = HNSW_MAGIC_NUMBER;
        uint32_t format_version = 1;
        out.write(reinterpret_cast< const char* >(&magic), sizeof(magic));
        out.write(reinterpret_cast< const char* >(&format_version), sizeof(format_version));

        // Hyperparameters
        uint64_t M = static_cast< uint64_t >(index.M_);
        uint64_t M_max_0 = static_cast< uint64_t >(index.M_max_0_);
        uint64_t ef_construction = static_cast< uint64_t >(index.ef_construction_);
        uint64_t ef_search = static_cast< uint64_t >(index.ef_search_);
        double mult = index.mult_;

        out.write(reinterpret_cast< const char* >(&M), sizeof(M));
        out.write(reinterpret_cast< const char* >(&M_max_0), sizeof(M_max_0));
        out.write(reinterpret_cast< const char* >(&ef_construction), sizeof(ef_construction));
        out.write(reinterpret_cast< const char* >(&ef_search), sizeof(ef_search));
        out.write(reinterpret_cast< const char* >(&mult), sizeof(mult));

        // Graph metadata
        uint64_t entry_point_id = index.entry_point_id_;
        int32_t max_graph_level = static_cast< int32_t >(index.max_graph_level_);
        uint8_t is_empty = index.is_empty_ ? 1 : 0;
        uint64_t count = static_cast< uint64_t >(index.nodes_.size());

        uint32_t dim = 0;
        if (!index.nodes_.empty()) {
            dim = static_cast< uint32_t >(index.nodes_.begin()->second.data.size());
        }

        out.write(reinterpret_cast< const char* >(&entry_point_id), sizeof(entry_point_id));
        out.write(reinterpret_cast< const char* >(&max_graph_level), sizeof(max_graph_level));
        out.write(reinterpret_cast< const char* >(&is_empty), sizeof(is_empty));
        out.write(reinterpret_cast< const char* >(&count), sizeof(count));
        out.write(reinterpret_cast< const char* >(&dim), sizeof(dim));

        // Nodes
        for (const auto& pair : index.nodes_) {
            const auto& node = pair.second;
            uint64_t id = node.id;
            out.write(reinterpret_cast< const char* >(&id), sizeof(id));

            if (dim > 0) {
                out.write(reinterpret_cast< const char* >(node.data.data()), dim * sizeof(float));
            }

            uint32_t num_levels = static_cast< uint32_t >(node.neighbors.size());
            out.write(reinterpret_cast< const char* >(&num_levels), sizeof(num_levels));

            for (uint32_t l = 0; l < num_levels; ++l) {
                const auto& level_neighbors = node.neighbors[l];
                uint32_t num_neighbors = static_cast< uint32_t >(level_neighbors.size());
                out.write(reinterpret_cast< const char* >(&num_neighbors), sizeof(num_neighbors));
                if (num_neighbors > 0) {
                    out.write(reinterpret_cast< const char* >(level_neighbors.data()), num_neighbors * sizeof(VectorID));
                }
            }
        }

        out.close();
        return out.good();
    }

    bool Storage::load_hnsw_from_file(const std::string& filepath, HNSWIndex& index) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in.is_open()) return false;

        uint32_t magic = 0;
        uint32_t format_version = 0;
        in.read(reinterpret_cast< char* >(&magic), sizeof(magic));
        if (magic != HNSW_MAGIC_NUMBER) return false;

        in.read(reinterpret_cast< char* >(&format_version), sizeof(format_version));
        if (format_version != 1) return false;

        // Hyperparameters
        uint64_t M = 0;
        uint64_t M_max_0 = 0;
        uint64_t ef_construction = 0;
        uint64_t ef_search = 0;
        double mult = 0.0;

        in.read(reinterpret_cast< char* >(&M), sizeof(M));
        in.read(reinterpret_cast< char* >(&M_max_0), sizeof(M_max_0));
        in.read(reinterpret_cast< char* >(&ef_construction), sizeof(ef_construction));
        in.read(reinterpret_cast< char* >(&ef_search), sizeof(ef_search));
        in.read(reinterpret_cast< char* >(&mult), sizeof(mult));

        // Graph metadata
        uint64_t entry_point_id = 0;
        int32_t max_graph_level = -1;
        uint8_t is_empty_val = 0;
        uint64_t count = 0;
        uint32_t dim = 0;

        in.read(reinterpret_cast< char* >(&entry_point_id), sizeof(entry_point_id));
        in.read(reinterpret_cast< char* >(&max_graph_level), sizeof(max_graph_level));
        in.read(reinterpret_cast< char* >(&is_empty_val), sizeof(is_empty_val));
        in.read(reinterpret_cast< char* >(&count), sizeof(count));
        in.read(reinterpret_cast< char* >(&dim), sizeof(dim));

        if (!in.good()) return false;

        index.M_ = static_cast< size_t >(M);
        index.M_max_0_ = static_cast< size_t >(M_max_0);
        index.ef_construction_ = static_cast< size_t >(ef_construction);
        index.ef_search_ = static_cast< size_t >(ef_search);
        index.mult_ = mult;
        index.entry_point_id_ = entry_point_id;
        index.max_graph_level_ = max_graph_level;
        index.is_empty_ = (is_empty_val != 0);

        index.nodes_.clear();
        index.nodes_.reserve(count);

        for (uint64_t i = 0; i < count; ++i) {
            HNSWIndex::Node node;
            in.read(reinterpret_cast< char* >(&node.id), sizeof(node.id));

            if (dim > 0) {
                node.data.resize(dim);
                in.read(reinterpret_cast< char* >(node.data.data()), dim * sizeof(float));
            }

            uint32_t num_levels = 0;
            in.read(reinterpret_cast< char* >(&num_levels), sizeof(num_levels));
            node.neighbors.resize(num_levels);

            for (uint32_t l = 0; l < num_levels; ++l) {
                uint32_t num_neighbors = 0;
                in.read(reinterpret_cast< char* >(&num_neighbors), sizeof(num_neighbors));
                node.neighbors[l].resize(num_neighbors);
                if (num_neighbors > 0) {
                    in.read(reinterpret_cast< char* >(node.neighbors[l].data()), num_neighbors * sizeof(VectorID));
                }
            }

            if (!in.good()) return false;

            index.nodes_[node.id] = std::move(node);
        }

        in.close();
        return true;
    }

}