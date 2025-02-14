// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
