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
#include <string>
#include <vector>

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::DescribeSnapshot()
 */
class MILVUS_SDK_API DescribeSnapshotResponse {
 public:
    /**
     * @brief Constructor
     */
    DescribeSnapshotResponse() = default;

    /**
     * @brief Get the snapshot name.
     * @return the name.
     */
    const std::string&
    Name() const;

    /**
     * @brief Set the snapshot name.
     * @param [in] name the name.
     */
    void
    SetName(std::string name);

    /**
     * @brief Get the snapshot description.
     * @return the description.
     */
    const std::string&
    Description() const;

    /**
     * @brief Set the snapshot description.
     * @param [in] description the description.
     */
    void
    SetDescription(std::string description);

    /**
     * @brief Get the collection the snapshot belongs to.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set the collection the snapshot belongs to.
     * @param [in] collection_name the collection name.
     */
    void
    SetCollectionName(std::string collection_name);

    /**
     * @brief Get the partition names covered by the snapshot.
     * @return the partition names.
     */
    const std::vector<std::string>&
    PartitionNames() const;

    /**
     * @brief Set the partition names covered by the snapshot.
     * @param [in] partition_names the partition names.
     */
    void
    SetPartitionNames(std::vector<std::string>&& partition_names);

    /**
     * @brief Get the creation timestamp of the snapshot.
     * @return the create ts.
     */
    int64_t
    CreateTs() const;

    /**
     * @brief Set the creation timestamp of the snapshot.
     * @param [in] create_ts the create ts.
     */
    void
    SetCreateTs(int64_t create_ts);

    /**
     * @brief Get the S3 storage location of the snapshot.
     * @return the S3 location.
     */
    const std::string&
    S3Location() const;

    /**
     * @brief Set the S3 storage location of the snapshot.
     * @param [in] s3_location the S3 location.
     */
    void
    SetS3Location(std::string s3_location);

 private:
    std::string name_;
    std::string description_;
    std::string collection_name_;
    std::vector<std::string> partition_names_;
    int64_t create_ts_{0};
    std::string s3_location_;
};

}  // namespace milvus
