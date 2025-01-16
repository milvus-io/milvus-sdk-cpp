#pragma once

#include <string>

class NodeInfo {
public:
    NodeInfo(int64_t id, const std::string& addr, const std::string& host);

    int64_t GetNodeId() const;
    const std::string& GetAddress() const;
    const std::string& GetHostname() const;

private:
    int64_t node_id;
    std::string address;
    std::string hostname;
};
