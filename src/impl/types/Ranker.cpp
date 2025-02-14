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

#include "milvus/types/Ranker.h"

namespace milvus {

nlohmann::json
BaseRanker::Dict() const {
    nlohmann::json json;
    json["strategy"] = GetStrategy();
    json["params"] = GetParams();
    return json;
}

RRFRanker::RRFRanker(float k) : k_(k) {
}

std::map<std::string, std::string>
RRFRanker::GetParams() const {
    return {{"k", std::to_string(k_)}};
}

std::string
RRFRanker::GetStrategy() const {
    return "rrf";
}

WeightedRanker::WeightedRanker(std::vector<float> weights) : weights_(weights) {
}

std::map<std::string, std::string>
WeightedRanker::GetParams() const {
    std::string weights_str = "[";
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_str += std::to_string(weights_[i]);
        if (i < weights_.size() - 1) {
            weights_str += ", ";
        }
    }
    weights_str += "]";

    return {{"weights", weights_str}};
}

std::string
WeightedRanker::GetStrategy() const {
    return "weighted";
}

}  // namespace milvus
