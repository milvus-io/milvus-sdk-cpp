#pragma once

#include <string>
#include <vector>
#include <utility>

namespace milvus {

class DatabaseDesc {
public:
    DatabaseDesc();
    DatabaseDesc(const std::string& db_name, int64_t db_id, uint64_t created_timestamp,
                const std::vector<std::pair<std::string, std::string>>& properties);

    const std::string& GetDbName() const;
    int64_t GetDbID() const;
    uint64_t GetCreatedTimestamp() const;
    const std::vector<std::pair<std::string, std::string>>& GetProperties() const;

    void SetDbName(const std::string& db_name);
    void SetDbID(int64_t db_id);
    void SetCreatedTimestamp(uint64_t created_timestamp);
    void SetProperties(const std::vector<std::pair<std::string, std::string>>& properties);
    void AddProperty(const std::string& key, const std::string& value);

private:
    std::string db_name_;
    int64_t db_id_{0};
    uint64_t created_timestamp_{0};
    std::vector<std::pair<std::string, std::string>> properties_;
};

}  // namespace milvus
