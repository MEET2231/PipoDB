#include "vectordb/Storage.h"
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

}