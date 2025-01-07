#pragma once

#include <string>
#include <vector>

namespace milvus {

class ListAliasesResult {
public:
    ListAliasesResult();
    ListAliasesResult(const std::string& db_name, const std::string& collection_name,
                      const std::vector<std::string>& aliases);

    const std::string& GetDbName() const;
    const std::string& GetCollectionName() const;
    const std::vector<std::string>& GetAliases() const;

    void SetDbName(const std::string& db_name);
    void SetCollectionName(const std::string& collection_name);
    void SetAliases(const std::vector<std::string>& aliases);

private:
    std::string db_name_{"default"};
    std::string collection_name_;
    std::vector<std::string> aliases_;
};

}  // namespace milvus
