#include "milvus/types/DatabaseDesc.h"

namespace milvus {

DatabaseDesc::DatabaseDesc() = default;

DatabaseDesc::DatabaseDesc(const std::string& db_name, int64_t db_id, uint64_t created_timestamp,
                         const std::vector<std::pair<std::string, std::string>>& properties)
    : db_name_(db_name), db_id_(db_id), created_timestamp_(created_timestamp), properties_(properties) {}

const std::string& 
DatabaseDesc::GetDbName() const {
    return db_name_;
}

int64_t 
DatabaseDesc::GetDbID() const {
    return db_id_;
}

uint64_t 
DatabaseDesc::GetCreatedTimestamp() const {
    return created_timestamp_;
}

const std::vector<std::pair<std::string, std::string>>& 
DatabaseDesc::GetProperties() const {
    return properties_;
}

void 
DatabaseDesc::SetDbName(const std::string& db_name) {
    db_name_ = db_name;
}

void 
DatabaseDesc::SetDbID(int64_t db_id) {
    db_id_ = db_id;
}

void 
DatabaseDesc::SetCreatedTimestamp(uint64_t created_timestamp) {
    created_timestamp_ = created_timestamp;
}

void 
DatabaseDesc::SetProperties(const std::vector<std::pair<std::string, std::string>>& properties) {
    properties_ = properties;
}

void 
DatabaseDesc::AddProperty(const std::string& key, const std::string& value) {
    properties_.emplace_back(key, value);
}

}  // namespace milvus
