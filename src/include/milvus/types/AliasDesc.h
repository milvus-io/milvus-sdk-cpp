#pragma once

#include <string>

namespace milvus {

class AliasDesc {
public:
    AliasDesc();
    AliasDesc(const std::string& db_name, const std::string& alias, const std::string& collection_name);

    const std::string& GetDbName() const;
    const std::string& GetAlias() const;
    const std::string& GetCollectionName() const;

    void SetDbName(const std::string& db_name);
    void SetAlias(const std::string& alias);
    void SetCollectionName(const std::string& collection_name);

private:
    std::string db_name_{"default"};
    std::string alias_;
    std::string collection_name_;
};

}  // namespace milvus
