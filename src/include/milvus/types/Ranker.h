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

namespace milvus {

class BaseRanker {
 public:
    virtual ~BaseRanker() = default;
    virtual std::map<std::string, std::string>
    GetParams() const = 0;
    virtual std::string
    GetStrategy() const = 0;
    virtual nlohmann::json
    Dict() const;
};

class RRFRanker : public BaseRanker {
 public:
    explicit RRFRanker(float k = 60.0);
    std::map<std::string, std::string>
    GetParams() const override;
    std::string
    GetStrategy() const override;

 private:
    float k_;
};

class WeightedRanker : public BaseRanker {
 public:
    explicit WeightedRanker(std::vector<float> weights);
    std::map<std::string, std::string>
    GetParams() const override;
    std::string
    GetStrategy() const override;

 private:
    std::vector<float> weights_;
};

}  // namespace milvus
