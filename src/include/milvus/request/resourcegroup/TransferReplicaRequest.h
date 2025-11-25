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

#include <string>

namespace milvus {

/**
 * @brief Used by MilvusClientV2::TransferReplica().
 */
class TransferReplicaRequest {
 public:
    /**
     * @brief Constructor
     */
    TransferReplicaRequest() = default;

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
    TransferReplicaRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Name of the collection.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of the collection.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Set name of the collection.
     */
    TransferReplicaRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Get name of the source resource group.
     */
    const std::string&
    SourceGroup() const;

    /**
     * @brief Set name of the source resource group.
     */
    void
    SetSourceGroup(const std::string& source_group);

    /**
     * @brief Set name of the source resource group.
     */
    TransferReplicaRequest&
    WithSourceGroup(const std::string& source_group);

    /**
     *  Get name of the target resource group.
     */
    const std::string&
    TargetGroup() const;

    /**
     *  Set name of the target resource group.
     */
    void
    SetTargetGroup(const std::string& target_group);

    /**
     *  Set name of the target resource group.
     */
    TransferReplicaRequest&
    WithTargetGroup(const std::string& target_group);

    /**
     *  Get number of replicas to transfer.
     */
    int64_t
    NumReplicas() const;

    /**
     *  Set number of replicas to transfer.
     */
    void
    SetNumReplicas(int64_t num);

    /**
     *  Set number of replicas to transfer.
     */
    TransferReplicaRequest&
    WithNumReplicas(int64_t num);

 protected:
    std::string db_name_;
    std::string collection_name_;
    std::string source_group_;
    std::string target_group_;
    int64_t num_replicas_{1};
};

}  // namespace milvus
