#include "milvus/types/ResourceGroupConfig.h"

namespace milvus {

ResourceGroupConfig::ResourceGroupConfig(int req_node_num, int lim_node_num,
                                         const std::vector<std::string>& from,
                                         const std::vector<std::string>& to,
                                         const std::vector<std::pair<std::string, std::string>>& labels)
    : requests_node_num(req_node_num), limits_node_num(lim_node_num),
      transfer_from(from), transfer_to(to), node_labels(labels) {}

int ResourceGroupConfig::GetRequestsNodeNum() const {
    return requests_node_num;
}

void ResourceGroupConfig::SetRequestsNodeNum(int num) {
    requests_node_num = num;
}

int ResourceGroupConfig::GetLimitsNodeNum() const {
    return limits_node_num;
}

void ResourceGroupConfig::SetLimitsNodeNum(int num) {
    limits_node_num = num;
}

const std::vector<std::string>& ResourceGroupConfig::GetTransferFrom() const {
    return transfer_from;
}

void ResourceGroupConfig::SetTransferFrom(const std::vector<std::string>& from) {
    transfer_from = from;
}

const std::vector<std::string>& ResourceGroupConfig::GetTransferTo() const {
    return transfer_to;
}

void ResourceGroupConfig::SetTransferTo(const std::vector<std::string>& to) {
    transfer_to = to;
}

const std::vector<std::pair<std::string, std::string>>& ResourceGroupConfig::GetNodeLabels() const {
    return node_labels;
}

void ResourceGroupConfig::SetNodeLabels(const std::vector<std::pair<std::string, std::string>>& labels) {
    node_labels = labels;
}

}  // namespace milvus
