#pragma once

#include <string>
#include <vector>
#include <map>
#include "NodeInfo.h"
#include "ResourceGroupConfig.h"

namespace milvus {

class ResourceGroupDesc {
public:
    ResourceGroupDesc() = default;
    ResourceGroupDesc(const std::string& name, int32_t capacity, int32_t available_nodes,
                      const std::map<std::string, int32_t>& loaded_replicas,
                      const std::map<std::string, int32_t>& outgoing_nodes,
                      const std::map<std::string, int32_t>& incoming_nodes,
                      const ResourceGroupConfig& config,
                      const std::vector<NodeInfo>& nodes);

    const std::string& GetName() const;
    int32_t GetCapacity() const;
    int32_t GetNumAvailableNode() const;
    const std::map<std::string, int32_t>& GetNumLoadedReplica() const;
    const std::map<std::string, int32_t>& GetNumOutgoingNode() const;
    const std::map<std::string, int32_t>& GetNumIncomingNode() const;
    const ResourceGroupConfig& GetConfig() const;
    const std::vector<NodeInfo>& GetNodes() const;

private:
    std::string name;
    int32_t capacity;
    int32_t num_available_node;
    std::map<std::string, int32_t> num_loaded_replica;
    std::map<std::string, int32_t> num_outgoing_node;
    std::map<std::string, int32_t> num_incoming_node;
    ResourceGroupConfig config;
    std::vector<NodeInfo> nodes;
};

}  // namespace milvus