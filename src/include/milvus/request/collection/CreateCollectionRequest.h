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
#include "../../types/IndexParam.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::CreateCollection()
 * @par Example
 * @code
 * milvus::CollectionSchema schema("demo");
 * schema.AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "", true, false));
 * schema.AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR).WithDimension(8));
 * auto status = client->CreateCollection(milvus::CreateCollectionRequest()
 *                                            .WithCollectionName("demo")
 *                                            .WithCollectionSchema(std::make_shared<milvus::CollectionSchema>(schema)));
 * @endcode
 */
class MILVUS_SDK_API CreateCollectionRequest {
 public:
    /**
     * @brief Constructor
     */
    CreateCollectionRequest() = default;

    /**
     * @brief Database name in which the collection is created.
     * @return the database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name in which the collection is created.
     * @param [in] db_name the DB name.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name in which the collection is created.
     * @param [in] db_name the DB name.
     */
    CreateCollectionRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Name of the collection.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of the collection.
     * Note: due to history reason, the CollectionSchema also contains a collection name.
     * SetCollectionName() will override the collection name of CollectionSchema.
     * @param [in] collection_name the collection name.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Set name of the collection.
     * Note: due to history reason, the CollectionSchema also contains a collection name.
     * WithCollectionName() will override the collection name of CollectionSchema.
     * @param [in] collection_name the collection name.
     */
    CreateCollectionRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Description of the collection.
     * @return the description.
     */
    const std::string&
    Description() const;

    /**
     * @brief Set name of the collection.
     * Note: due to history reason, the CollectionSchema also contains a description.
     * SetDescription() will override description of CollectionSchema.
     * @param [in] description the description.
     */
    void
    SetDescription(const std::string& description);

    /**
     * @brief Set name of the collection.
     * Note: due to history reason, the CollectionSchema also contains a collection name.
     * WithDescription() will override description of CollectionSchema.
     * @param [in] description the description.
     */
    CreateCollectionRequest&
    WithDescription(const std::string& description);

    /**
     * @brief Collection schema.
     * @return the collection schema.
     */
    const CollectionSchemaPtr&
    CollectionSchema() const;

    /**
     * @brief Set collection schema.
     * @param [in] schema the schema.
     */
    void
    SetCollectionSchema(const CollectionSchemaPtr& schema);

    /**
     * @brief Set collection schema.
     * @param [in] schema the schema.
     */
    CreateCollectionRequest&
    WithCollectionSchema(const CollectionSchemaPtr& schema);

    /**
     * @brief Get number of partitions when there is a partition key.
     * @return the num partitions.
     */
    int64_t
    NumPartitions() const;

    /**
     * @brief Set number of partitions when there is a partition key.
     * @param [in] num_partitions the num partitions.
     */
    void
    SetNumPartitions(int64_t num_partitions);

    /**
     * @brief Set number of partitions when there is a partition key.
     * @param [in] num_partitions the num partitions.
     */
    CreateCollectionRequest&
    WithNumPartitions(int64_t num_partitions);

    /**
     * @brief Get number of shards of the collection.
     * @return the num shards.
     */
    int64_t
    NumShards() const;

    /**
     * @brief Set number of shards of the collection.
     * Note: due to history reason, the CollectionSchema also contains a shards number.
     * SetNumShards() will override the shards_num of CollectionSchema.
     * @param [in] num_shards the num shards.
     */
    void
    SetNumShards(int64_t num_shards);

    /**
     * @brief Set number of shards of the collection.
     * Note: due to history reason, the CollectionSchema also contains a shards number.
     * WithNumShards() will override the shards_num of CollectionSchema.
     * @param [in] num_shards the num shards.
     */
    CreateCollectionRequest&
    WithNumShards(int64_t num_shards);

    /**
     * @brief Get default consistency level of this collection.
     * @return the consistency level.
     */
    ConsistencyLevel
    GetConsistencyLevel() const;

    /**
     * @brief Set default consistency level of this collection.
     * @param [in] level the level.
     */
    void
    SetConsistencyLevel(ConsistencyLevel level);

    /**
     * @brief Set default consistency level of this collection.
     * @param [in] level the level.
     */
    CreateCollectionRequest&
    WithConsistencyLevel(ConsistencyLevel level);

    /**
     * @brief Get properties of this collection.
     *
     * For example, "collection.ttl.seconds" sets a collection-level TTL retention window.
     * @return the properties.
     */
    const std::unordered_map<std::string, std::string>&
    Properties() const;

    /**
     * @brief Set properties of this collection.
     *
     * For example, "collection.ttl.seconds" sets a collection-level TTL retention window.
     * @param [in] properties the properties.
     */
    void
    SetProperties(std::unordered_map<std::string, std::string>&& properties);

    /**
     * @brief Set properties of this collection.
     * @param [in] properties the properties.
     */
    CreateCollectionRequest&
    WithProperties(std::unordered_map<std::string, std::string>&& properties);

    /**
     * @brief Set a property of this collection.
     * @param [in] key the key.
     * @param [in] property the property.
     */
    CreateCollectionRequest&
    AddProperty(const std::string& key, const std::string& property);

    /**
     * @brief Get index params.
     * @return the index params.
     */
    const std::vector<IndexParam>&
    IndexParams() const;

    /**
     * @brief Set index params to be created.
     * @param [in] index_params the index params.
     */
    void
    SetIndexParams(std::vector<IndexParam>&& index_params);

    /**
     * @brief Set index params to be created.
     * @param [in] index_params the index params.
     */
    CreateCollectionRequest&
    WithIndexParams(std::vector<IndexParam>&& index_params);

    /**
     * @brief Add an index param to be created.
     * @param [in] index_param the index param.
     */
    CreateCollectionRequest&
    AddIndexParam(IndexParam&& index_param);

    /**
     * @brief Get indexes.
     * @return the indexes.
     * @deprecated Use IndexParams() instead.
     */
    [[deprecated("use IndexParams() instead")]] const std::vector<IndexDesc>&
    Indexes() const;

    /**
     * @brief Set indexes to be created.
     * @param [in] indexes the indexes.
     * @deprecated Use WithIndexParams() instead.
     */
    [[deprecated("use WithIndexParams() instead")]] void
    SetIndexes(std::vector<IndexDesc>&& indexes);

    /**
     * @brief Set indexes to be created.
     * @param [in] indexes the indexes.
     * @deprecated Use WithIndexParams() instead.
     */
    [[deprecated("use WithIndexParams() instead")]] CreateCollectionRequest&
    WithIndexes(std::vector<IndexDesc>&& indexes);

    /**
     * @brief Add an index to be created.
     * @param [in] index the index.
     * @deprecated Use AddIndexParam() instead.
     */
    [[deprecated("use AddIndexParam() instead")]] CreateCollectionRequest&
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
    std::vector<IndexParam> index_params_;
    mutable std::vector<IndexDesc> indexes_cache_;
};

}  // namespace milvus
