#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "../Status.h"

namespace milvus {

/**
 * @brief Arguments for MilvusClientV2::Get().
 */
class GetArguments {
 public:
    /**
     * @brief Get name of the target collection
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of this collection, cannot be empty
     */
    Status
    SetCollectionName(std::string collection_name);

    /**
     * @brief Get partition names
     */
    const std::set<std::string>&
    PartitionNames() const;

    /**
     * @brief Specify partition name to control get scope, the name cannot be empty
     */
    Status
    AddPartitionName(std::string partition_name);

    /**
     * @brief Get output field names
     */
    const std::set<std::string>&
    OutputFields() const;

    /**
     * @brief Specify output field names to return field data, the name cannot be empty
     */
    Status
    AddOutputField(std::string field_name);

    /**
     * @brief Get primary key ids
     */
    const std::vector<int64_t>&
    Ids() const;

    /**
     * @brief Set primary key ids to get
     */
    Status
    SetIds(std::vector<int64_t> ids);

 private:
    std::string collection_name_;
    std::set<std::string> partition_names_;
    std::set<std::string> output_field_names_;
    std::vector<int64_t> ids_;
};

}  // namespace milvus 