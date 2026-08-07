#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "vectordb/Vector.h"

namespace vectordb {

    class Storage {
    private:
        static constexpr uint32_t MAGIC_NUMBER = 0x5049504F; 

    public:
        static bool save_to_file(const std::string& filepath, const std::vector< Vector >& vectors);
        static bool load_from_file(const std::string& filepath, std::vector< Vector >& out_vectors);
    };

}