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
#include <unordered_map>

namespace milvus {

/**
 * @brief Collection statistics returned by MilvusClient::GetCollectionStats().
 */
class CollectionStat {
 public:
    /**
     * @brief Construct a new Collection Stat object
     */
    CollectionStat();

    /**
     * @brief Return row count of this collection.
     *
     */
    uint64_t
    RowCount() const;

    /**
     * @brief Set collection name
     *
     */
    void
    SetName(std::string name);

    /**
     * @brief Get collection name
     *
     */
    const std::string&
    Name() const;

    /**
     * @brief add key/value pair for collection statistics
     */
    void
    Emplace(std::string key, std::string value);

 private:
    std::string name_;
    std::unordered_map<std::string, std::string> statistics_;
};

}  // namespace milvus
