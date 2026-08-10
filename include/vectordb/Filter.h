#pragma once

#include <string>
#include <vector>

namespace vectordb {

    enum class FilterOp {
        EQ,       // Equals (==)
        NEQ,      // Not Equals (!=)
        GT,       // Greater Than (>)
        GTE,      // Greater Than or Equal (>=)
        LT,       // Less Than (<)
        LTE,      // Less Than or Equal (<=)
        CONTAINS  // Substring match
    };

    struct FilterCondition {
        std::string key;        // Field name in JSON payload (e.g., "category", "year")
        FilterOp op;            // Comparison operator
        std::string value;      // Comparison value (e.g., "tech", "2024", "true")
    };

    class PayloadFilter {
    public:
        std::vector<FilterCondition> must;    // All conditions must match (AND logic)
        std::vector<FilterCondition> should;  // At least one condition must match (OR logic)

        bool empty() const { return must.empty() && should.empty(); }

        // Evaluates if a JSON payload string satisfies all filter conditions
        bool matches(const std::string& payload_json) const;
    };

}
