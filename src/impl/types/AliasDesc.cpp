#include "milvus/types/AliasDesc.h"

namespace milvus {

AliasDesc::AliasDesc() = default;

AliasDesc::AliasDesc(const std::string& db_name, const std::string& alias, const std::string& collection_name)
    : db_name_(db_name), alias_(alias), collection_name_(collection_name) {}

const std::string& 
AliasDesc::GetDbName() const {
    return db_name_;
}

const std::string& 
AliasDesc::GetAlias() const {
    return alias_;
}

const std::string& 
AliasDesc::GetCollectionName() const {
    return collection_name_;
}

void 
AliasDesc::SetDbName(const std::string& db_name) {
    db_name_ = db_name;
}

void 
AliasDesc::SetAlias(const std::string& alias) {
    alias_ = alias;
}

void 
AliasDesc::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

}  // namespace milvus
