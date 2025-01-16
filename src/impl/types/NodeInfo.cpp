#include "milvus/types/NodeInfo.h"

NodeInfo::NodeInfo(int64_t id, const std::string& addr, const std::string& host)
    : node_id(id), address(addr), hostname(host) {}

int64_t NodeInfo::GetNodeId() const {
    return node_id;
}

const std::string& NodeInfo::GetAddress() const {
    return address;
}

const std::string& NodeInfo::GetHostname() const {
    return hostname;
}
