#include "milvus/types/ResourceGroupDesc.h"

namespace milvus {

ResourceGroupDesc::ResourceGroupDesc(const std::string& name, int32_t capacity, int32_t available_nodes,
                                     const std::map<std::string, int32_t>& loaded_replicas,
                                     const std::map<std::string, int32_t>& outgoing_nodes,
                                     const std::map<std::string, int32_t>& incoming_nodes,
                                     const ResourceGroupConfig& config,
                                     const std::vector<NodeInfo>& nodes)
    : name(name), capacity(capacity), num_available_node(available_nodes),
      num_loaded_replica(loaded_replicas), num_outgoing_node(outgoing_nodes),
      num_incoming_node(incoming_nodes), config(config), nodes(nodes) {}

const std::string& ResourceGroupDesc::GetName() const {
    return name;
}

int32_t ResourceGroupDesc::GetCapacity() const {
    return capacity;
}

int32_t ResourceGroupDesc::GetNumAvailableNode() const {
    return num_available_node;
}

const std::map<std::string, int32_t>& ResourceGroupDesc::GetNumLoadedReplica() const {
    return num_loaded_replica;
}

const std::map<std::string, int32_t>& ResourceGroupDesc::GetNumOutgoingNode() const {
    return num_outgoing_node;
}

const std::map<std::string, int32_t>& ResourceGroupDesc::GetNumIncomingNode() const {
    return num_incoming_node;
}

const ResourceGroupConfig& ResourceGroupDesc::GetConfig() const {
    return config;
}

const std::vector<NodeInfo>& ResourceGroupDesc::GetNodes() const {
    return nodes;
}

}  // namespace milvus
