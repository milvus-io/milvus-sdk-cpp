#pragma once

#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "FieldData.h"

namespace milvus {

class AnnSearchRequest {
 public:
    AnnSearchRequest(const std::string& anns_field, const std::map<std::string, std::string>& param, int limit,
                     const std::string& expr = "")
        : anns_field_(anns_field), param_(param), limit_(limit), expr_(expr) {
    }

    const std::string&
    AnnsField() const;
    const std::map<std::string, std::string>&
    Param() const;
    int
    Limit() const;
    const std::string&
    Expr() const;

    FieldDataPtr
    TargetVectors() const;

    Status
    AddTargetVector(std::string field_name, const std::string& vector);

    Status
    AddTargetVector(std::string field_name, const std::vector<uint8_t>& vector);

    Status
    AddTargetVector(std::string field_name, std::string&& vector);

    Status
    AddTargetVector(std::string field_name, const FloatVecFieldData::ElementT& vector);

    Status
    AddTargetVector(std::string field_name, FloatVecFieldData::ElementT&& vector);

 private:
    std::string anns_field_;
    std::map<std::string, std::string> param_;
    int limit_;
    std::string expr_;

    BinaryVecFieldDataPtr binary_vectors_;
    FloatVecFieldDataPtr float_vectors_;
};

}  // namespace milvus
