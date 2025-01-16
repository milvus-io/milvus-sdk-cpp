#pragma once

#include <string>
#include <vector>
#include <utility>

namespace milvus {

class ResourceGroupConfig {
public:
    ResourceGroupConfig() = default;
    ResourceGroupConfig(int req_node_num, int lim_node_num,
                        const std::vector<std::string>& from,
                        const std::vector<std::string>& to,
                        const std::vector<std::pair<std::string, std::string>>& labels);

    int GetRequestsNodeNum() const;
    void SetRequestsNodeNum(int num);

    int GetLimitsNodeNum() const;
    void SetLimitsNodeNum(int num);

    const std::vector<std::string>& GetTransferFrom() const;
    void SetTransferFrom(const std::vector<std::string>& from);

    const std::vector<std::string>& GetTransferTo() const;
    void SetTransferTo(const std::vector<std::string>& to);

    const std::vector<std::pair<std::string, std::string>>& GetNodeLabels() const;
    void SetNodeLabels(const std::vector<std::pair<std::string, std::string>>& labels);

private:
    int requests_node_num;
    int limits_node_num;
    std::vector<std::string> transfer_from;
    std::vector<std::string> transfer_to;
    std::vector<std::pair<std::string, std::string>> node_labels;
};

} // namespace milvus
