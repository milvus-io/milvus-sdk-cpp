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
 * @brief Used by MilvusClientV2::Compact()
 */
class CompactRequest {
 public:
    /**
     * @brief Constructor
     */
    CompactRequest() = default;

    /**
     * @brief Database name in which the collections are created.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name in which the collections are created.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name in which the collections are created.
     */
    CompactRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Name of the collection to be compacted.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of the collection to be compacted.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Set name of the collection to be compacted.
     */
    CompactRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Get the flag whether it is cluserting compaction or not.
     */
    bool
    ClusteringCompaction() const;

    /**
     * @brief Set cluserting compaction flag.
     * True: do cluserting compaction, report error if no clustering key.
     * False: do normal compaction.
     */
    void
    SetClusteringCompaction(bool clustering_compaction);

    /**
     * @brief Set cluserting compaction flag.
     * True: do cluserting compaction, report error if no clustering key.
     * False: do normal compaction.
     */
    CompactRequest&
    WithClusteringCompaction(bool clustering_compaction);

 private:
    std::string db_name_;
    std::string collection_name_;
    bool is_clustring_compaction_{false};
};

}  // namespace milvus
