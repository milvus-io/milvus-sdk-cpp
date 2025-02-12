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
