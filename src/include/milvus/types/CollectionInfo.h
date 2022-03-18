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

#include <memory>
#include <string>
#include <vector>

namespace milvus {

/**
 * @brief Collection runtime information including create timestamp and loading percentage, returned by
 * MilvusClient::ShowCollections().
 */
class CollectionInfo {
 public:
    /**
     * @brief Constructor
     */
    CollectionInfo();

    /**
     * @brief Constructor
     */
    CollectionInfo(const std::string& collection_name, int64_t collection_id, uint64_t create_time,
                   uint64_t load_percentage);

    /**
     * @brief Construct a new Collection Info object
     */
    CollectionInfo(CollectionInfo&&) noexcept;

    /**
     * @brief Destroy the Collection Info object
     */
    ~CollectionInfo();

    /**
     * @brief Name of the collection.
     */
    const std::string&
    Name() const;

    /**
     * @brief Internal ID of the collection.
     */
    int64_t
    ID() const;

    /**
     * @brief The utc time when the collection is created.
     */
    uint64_t
    CreatedTime() const;

    /**
     * @brief Loading percentage of the collection.
     */
    uint64_t
    MemoryPercentage() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief CollectionsInfo objects array
 */
using CollectionsInfo = std::vector<CollectionInfo>;

}  // namespace milvus
