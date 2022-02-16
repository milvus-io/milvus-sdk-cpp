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
#include <vector>

#include "CollectionSchema.h"

namespace milvus {

/**
 * @brief Collection schema and runtime information returned by MilvusClient::DescribeCollection().
 */
class CollectionDesc {
 public:
    CollectionDesc() = default;

    /**
     * @brief Collection schema.
     */
    CollectionSchema
    Schema() const {
        return schema_;
    }

    /**
     * @brief Set collection schema.
     */
    void
    SetSchema(CollectionSchema&& schema) {
        schema_ = std::move(schema);
    }

    /**
     * @brief Collection id.
     */
    int64_t
    ID() const {
        return collection_id_;
    }

    /**
     * @brief Set collection id.
     */
    void
    SetID(const int64_t id) {
        collection_id_ = id;
    }

    /**
     * @brief Collection alias.
     */
    const std::vector<std::string>&
    Alias() const {
        return alias_;
    }

    /**
     * @brief Set collection alias.
     */
    void
    SetAlias(std::vector<std::string>&& alias) {
        alias_ = std::move(alias);
    }

    /**
     * @brief Timestamp when the collection created.
     */
    uint64_t
    CreatedTime() const {
        return created_utc_timestamp_;
    }

    /**
     * @brief Set timestamp when the collection created.
     */
    void
    SetCreatedTime(const uint64_t ts) {
        created_utc_timestamp_ = ts;
    }

 private:
    CollectionSchema schema_;

    int64_t collection_id_;
    std::vector<std::string> alias_;
    uint64_t created_utc_timestamp_ = 0;
};

}  // namespace milvus
