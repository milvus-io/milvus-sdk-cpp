// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <memory>
#include <milvus/thirdparty/nlohmann/json.hpp>
#include <string>
#include <vector>

#include "CollectionSchema.h"
#include "ConsistencyLevel.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Collection schema and runtime information returned by MilvusClient::DescribeCollection().
 */
class MILVUS_SDK_API CollectionDesc {
 public:
    /**
     * @brief The database name which this collection belong to.
     * @return the database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name.
     * @param [in] name the name.
     */
    void
    SetDatabaseName(std::string name);

    /**
     * @brief The collection name.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Description of the collection.
     * @return the description.
     */
    const std::string&
    Description() const;

    /**
     * @brief Shards number of the collection.
     * @return the num shards.
     */
    int64_t
    NumShards() const;

    /**
     * @brief Collection schema.
     * @return the schema.
     */
    const CollectionSchema&
    Schema() const;

    /**
     * @brief Set collection schema.
     * @param [in] schema the schema.
     */
    void
    SetSchema(const CollectionSchema& schema);

    /**
     * @brief Set collection schema.
     * @param [in] schema the schema.
     */
    void
    SetSchema(CollectionSchema&& schema);

    /**
     * @brief Collection id.
     * @return the ID.
     */
    int64_t
    ID() const;

    /**
     * @brief Set collection id.
     * @param [in] id the ID.
     */
    void
    SetID(int64_t id);

    /**
     * @brief Collection alias.
     * @return the alias.
     */
    const std::vector<std::string>&
    Alias() const;

    /**
     * @brief Set collection alias.
     * @param [in] alias the alias.
     */
    void
    SetAlias(const std::vector<std::string>& alias);

    /**
     * @brief Set collection alias.
     * @param [in] alias the alias.
     */
    void
    SetAlias(std::vector<std::string>&& alias);

    /**
     * @brief Timestamp when the collection created.
     * @return the created time.
     */
    uint64_t
    CreatedTime() const;

    /**
     * @brief Set timestamp when the collection created.
     * @param [in] ts the ts.
     */
    void
    SetCreatedTime(uint64_t ts);

    /**
     * @brief Timestamp when the collection is updated.
     * @return the update time.
     */
    uint64_t
    UpdateTime() const;

    /**
     * @brief Set timestamp when the collection is updated.
     * @param [in] ts the ts.
     */
    void
    SetUpdateTime(uint64_t ts);

    /**
     * @brief Collection properties.
     * @return the properties.
     */
    const std::unordered_map<std::string, std::string>&
    Properties() const;

    /**
     * @brief Set properties of the collection.
     * @param [in] properties the properties.
     */
    void
    SetProperties(std::unordered_map<std::string, std::string>&& properties);

    /**
     * @brief Get the external data source of the collection.
     * @return the external source.
     */
    const std::string&
    ExternalSource() const;

    /**
     * @brief Set the external data source of the collection.
     *
     * @param [in] external_source
     */
    void
    SetExternalSource(std::string external_source);

    /**
     * @brief Get the external file specification of the collection.
     * @return the external spec.
     */
    const nlohmann::json&
    ExternalSpec() const;

    /**
     * @brief Set the external file specification of the collection.
     *
     * @param [in] external_spec
     */
    void
    SetExternalSpec(const nlohmann::json& external_spec);

    /**
     * @brief Consistency level of the collection.
     * @return the consistency level.
     */
    ConsistencyLevel
    GetConsistencyLevel() const;

    /**
     * @brief Set consistency level of the collection.
     * @param [in] level the level.
     */
    void
    SetConsistencyLevel(ConsistencyLevel level);

    /**
     * @brief Number of partitions of the collection.
     * Only valid when the collection is created with a partition key.
     * @return the num partitions.
     */
    int64_t
    NumPartitions() const;

    /**
     * @brief Set number of partitions of the collection.
     * @param [in] num_partitions the num partitions.
     */
    void
    SetNumPartitions(int64_t num_partitions);

 private:
    std::string db_name_;
    CollectionSchema schema_;
    int64_t collection_id_;
    std::vector<std::string> alias_;
    uint64_t created_utc_timestamp_ = 0;
    uint64_t update_timestamp_ = 0;
    std::unordered_map<std::string, std::string> properties_;
    std::string external_source_;
    nlohmann::json external_spec_;
    ConsistencyLevel consistency_level_{ConsistencyLevel::BOUNDED};
    int64_t num_partitions_{0};
};

using CollectionDescPtr = std::shared_ptr<CollectionDesc>;
}  // namespace milvus
