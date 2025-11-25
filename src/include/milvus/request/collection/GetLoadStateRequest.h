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

#include <set>

#include "./CollectionRequestBase.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::GetLoadState()
 */
class GetLoadStateRequest : public CollectionRequestBase {
 public:
    /**
     * @brief Constructor
     */
    GetLoadStateRequest() = default;

    /**
     * @brief Set database name in which the collection is created.
     */
    GetLoadStateRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Set name of the collection.
     */
    GetLoadStateRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Get partition names to get load state.
     * If partition name list is empty, will get load state of the collection.
     */
    const std::set<std::string>&
    PartitionNames() const;

    /**
     * @brief Set partition names to get load state of the partition.
     * If partition name list is empty, will get load state of the collection.
     */
    void
    SetPartitionNames(std::set<std::string>&& partition_names);

    /**
     * @brief Set partition names to get load state of the partition.
     * If partition name list is empty, will get load state of the collection.
     */
    GetLoadStateRequest&
    WithPartitionNames(std::set<std::string>&& partition_names);

    /**
     * @brief Add a partition name to get load state.
     */
    GetLoadStateRequest&
    AddPartitionName(const std::string& partition_name);

 private:
    std::set<std::string> partition_names_;
};

}  // namespace milvus
