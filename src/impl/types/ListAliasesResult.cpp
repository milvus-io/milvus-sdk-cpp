#include "milvus/types/ListAliasesResult.h"

namespace milvus {

ListAliasesResult::ListAliasesResult() = default;

ListAliasesResult::ListAliasesResult(const std::string& db_name, const std::string& collection_name,
                                   const std::vector<std::string>& aliases)
    : db_name_(db_name), collection_name_(collection_name), aliases_(aliases) {}

const std::string& 
ListAliasesResult::GetDbName() const {
    return db_name_;
}

const std::string& 
ListAliasesResult::GetCollectionName() const {
    return collection_name_;
}

const std::vector<std::string>& 
ListAliasesResult::GetAliases() const {
    return aliases_;
}

void 
ListAliasesResult::SetDbName(const std::string& db_name) {
    db_name_ = db_name;
}

void 
ListAliasesResult::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

void 
ListAliasesResult::SetAliases(const std::vector<std::string>& aliases) {
    aliases_ = aliases;
}

}  // namespace milvus
