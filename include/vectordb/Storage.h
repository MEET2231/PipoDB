#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "vectordb/Vector.h"

namespace vectordb {

    class HNSWIndex;

    class Storage {
    private:
        static constexpr uint32_t MAGIC_NUMBER = 0x5049504F;      // Binary format for FlatIndex ("PIPO")
        static constexpr uint32_t HNSW_MAGIC_NUMBER = 0x484E5357; // Binary format for HNSWIndex ("HNSW")

    public:
        static bool save_to_file(const std::string& filepath, const std::vector< Vector >& vectors);
        static bool load_from_file(const std::string& filepath, std::vector< Vector >& out_vectors);

        static bool save_hnsw_to_file(const std::string& filepath, const HNSWIndex& index);
        static bool load_hnsw_from_file(const std::string& filepath, HNSWIndex& index);
    };

}