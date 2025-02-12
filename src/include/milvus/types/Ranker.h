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
