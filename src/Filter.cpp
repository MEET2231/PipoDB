#include "vectordb/Filter.h"
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <iostream>

namespace vectordb {

    namespace {
        std::string trim(const std::string& str) {
            size_t first = str.find_first_not_of(" \t\n\r\"'");
            if (first == std::string::npos) return "";
            size_t last = str.find_last_not_of(" \t\n\r\"',}");
            return str.substr(first, (last - first + 1));
        }

        std::string extract_json_value(const std::string& json, const std::string& key) {
            std::string search_key = "\"" + key + "\"";
            size_t key_pos = json.find(search_key);
            if (key_pos == std::string::npos) {
                // Try key without quotes
                search_key = key;
                key_pos = json.find(search_key);
                if (key_pos == std::string::npos) return "";
            }

            size_t colon_pos = json.find(':', key_pos + search_key.length());
            if (colon_pos == std::string::npos) return "";

            size_t val_start = colon_pos + 1;
            while (val_start < json.length() && (json[val_start] == ' ' || json[val_start] == '\t')) {
                val_start++;
            }

            if (val_start >= json.length()) return "";

            size_t val_end = json.find_first_of(",}\n\r", val_start);
            if (val_end == std::string::npos) val_end = json.length();

            std::string raw_val = json.substr(val_start, val_end - val_start);
            return trim(raw_val);
        }

        bool eval_condition(const std::string& json_val, FilterOp op, const std::string& target_val) {
            if (json_val.empty()) return false;

            if (op == FilterOp::EQ) {
                return json_val == target_val;
            } else if (op == FilterOp::NEQ) {
                return json_val != target_val;
            } else if (op == FilterOp::CONTAINS) {
                return json_val.find(target_val) != std::string::npos;
            } else {
                // Numeric comparison
                try {
                    double num_json = std::stod(json_val);
                    double num_target = std::stod(target_val);

                    switch (op) {
                        case FilterOp::GT:  return num_json > num_target;
                        case FilterOp::GTE: return num_json >= num_target;
                        case FilterOp::LT:  return num_json < num_target;
                        case FilterOp::LTE: return num_json <= num_target;
                        default: return false;
                    }
                } catch (...) {
                    return false;
                }
            }
        }
    }

    bool PayloadFilter::matches(const std::string& payload_json) const {
        if (empty()) return true;
        if (payload_json.empty()) return false;

        // 1. Evaluate MUST conditions (AND logic)
        for (const auto& cond : must) {
            std::string actual_val = extract_json_value(payload_json, cond.key);
            if (!eval_condition(actual_val, cond.op, cond.value)) {
                return false;
            }
        }

        // 2. Evaluate SHOULD conditions (OR logic)
        if (!should.empty()) {
            bool any_matched = false;
            for (const auto& cond : should) {
                std::string actual_val = extract_json_value(payload_json, cond.key);
                if (eval_condition(actual_val, cond.op, cond.value)) {
                    any_matched = true;
                    break;
                }
            }
            if (!any_matched) return false;
        }

        return true;
    }

}
