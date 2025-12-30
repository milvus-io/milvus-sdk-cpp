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

#include <unordered_map>

#include "../../types/CollectionSchema.h"
#include "../../types/ConsistencyLevel.h"
#include "../../types/IndexDesc.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::CreateCollection()
 */
class CreateCollectionRequest {
 public:
    /**
     * @brief Constructor
     */
    CreateCollectionRequest() = default;

    /**
     * @brief Database name in which the collection is created.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name in which the collection is created.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name in which the collection is created.
     */
    CreateCollectionRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Name of the collection.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of the collection.
     * Note: due to history reason, the CollectionSchema also contains a collection name.
     * SetCollectionName() will override the collection name of CollectionSchema.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Set name of the collection.
     * Note: due to history reason, the CollectionSchema also contains a collection name.
     * WithCollectionName() will override the collection name of CollectionSchema.
     */
    CreateCollectionRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Description of the collection.
     */
    const std::string&
    Description() const;

    /**
     * @brief Set name of the collection.
     * Note: due to history reason, the CollectionSchema also contains a description.
     * SetDescription() will override description of CollectionSchema.
     */
    void
    SetDescription(const std::string& description);

    /**
     * @brief Set name of the collection.
     * Note: due to history reason, the CollectionSchema also contains a collection name.
     * WithDescription() will override description of CollectionSchema.
     */
    CreateCollectionRequest&
    WithDescription(const std::string& description);

    /**
     * @brief Collection schema.
     */
    const CollectionSchemaPtr&
    CollectionSchema() const;

    /**
     * @brief Set collection schema.
     */
    void
    SetCollectionSchema(const CollectionSchemaPtr& schema);

    /**
     * @brief Set collection schema.
     */
    CreateCollectionRequest&
    WithCollectionSchema(const CollectionSchemaPtr& schema);

    /**
     * @brief Get number of partitions when there is a partition key.
     */
    int64_t
    NumPartitions() const;

    /**
     * @brief Set number of partitions when there is a partition key.
     */
    void
    SetNumPartitions(int64_t num_partitions);

    /**
     * @brief Set number of partitions when there is a partition key.
     */
    CreateCollectionRequest&
    WithNumPartitions(int64_t num_partitions);

    /**
     * @brief Get number of shards of the collection.
     */
    int64_t
    NumShards() const;

    /**
     * @brief Set number of shards of the collection.
     * Note: due to history reason, the CollectionSchema also contains a shards number.
     * SetNumShards() will override the shards_num of CollectionSchema.
     */
    void
    SetNumShards(int64_t num_shards);

    /**
     * @brief Set number of shards of the collection.
     * Note: due to history reason, the CollectionSchema also contains a shards number.
     * WithNumShards() will override the shards_num of CollectionSchema.
     */
    CreateCollectionRequest&
    WithNumShards(int64_t num_shards);

    /**
     * @brief Get default consistency level of this collection.
     */
    ConsistencyLevel
    GetConsistencyLevel() const;

    /**
     * @brief Set default consistency level of this collection.
     */
    void
    SetConsistencyLevel(ConsistencyLevel level);

    /**
     * @brief Set default consistency level of this collection.
     */
    CreateCollectionRequest&
    WithConsistencyLevel(ConsistencyLevel level);

    /**
     * @brief Get properties of this collection.
     */
    const std::unordered_map<std::string, std::string>&
    Properties() const;

    /**
     * @brief Set properties of this collection.
     */
    void
    SetProperties(std::unordered_map<std::string, std::string>&& properties);

    /**
     * @brief Set properties of this collection.
     */
    CreateCollectionRequest&
    WithProperties(std::unordered_map<std::string, std::string>&& properties);

    /**
     * @brief Set a property of this collection.
     */
    CreateCollectionRequest&
    AddProperty(const std::string& key, const std::string& property);

    /**
     * @brief Get indexes.
     */
    const std::vector<IndexDesc>&
    Indexes() const;

    /**
     * @brief Set indexes to be created.
     */
    void
    SetIndexes(std::vector<IndexDesc>&& indexes);

    /**
     * @brief Set indexes to be created.
     */
    CreateCollectionRequest&
    WithIndexes(std::vector<IndexDesc>&& indexes);

    /**
     * @brief Add an index to be created.
     */
    CreateCollectionRequest&
    AddIndex(IndexDesc&& index);

 private:
    std::string db_name_;
    std::string collection_name_;
    std::string description_;
    CollectionSchemaPtr schema_;
    int64_t num_partitions_{0};
    int64_t num_shards_{1};
    ConsistencyLevel level_{ConsistencyLevel::BOUNDED};
    std::unordered_map<std::string, std::string> properties_;
    std::vector<IndexDesc> indexes_;
};

}  // namespace milvus
