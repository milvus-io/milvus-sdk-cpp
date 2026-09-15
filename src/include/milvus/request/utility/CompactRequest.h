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

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Compact()
 */
class MILVUS_SDK_API CompactRequest {
 public:
    /**
     * @brief Constructor
     */
    CompactRequest() = default;

    /**
     * @brief Database name in which the collections are created.
     * @return the database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name in which the collections are created.
     * @param [in] db_name the DB name.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name in which the collections are created.
     * @param [in] db_name the DB name.
     */
    CompactRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Name of the collection to be compacted.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of the collection to be compacted.
     * @param [in] collection_name the collection name.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Set name of the collection to be compacted.
     * @param [in] collection_name the collection name.
     */
    CompactRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Target segment size, expressed in the unit returned by TargetSizeUnit().
     * Zero means use server default. Must be a positive integer when set.
     * @return the target size.
     */
    int64_t
    TargetSize() const;

    /**
     * @brief Set target segment size, expressed in the unit returned by TargetSizeUnit().
     * Zero means use server default.
     * @param [in] target_size the target size.
     */
    void
    SetTargetSize(int64_t target_size);

    /**
     * @brief Set target segment size, expressed in the unit returned by TargetSizeUnit().
     * Zero means use server default.
     * @param [in] target_size the target size.
     */
    CompactRequest&
    WithTargetSize(int64_t target_size);

    /**
     * @brief Unit of the target segment size. Supported values: "b", "kb", "mb", "gb",
     * "tb", "pb". Default is "mb".
     * @return the target size unit.
     */
    const std::string&
    TargetSizeUnit() const;

    /**
     * @brief Set unit of the target segment size. Supported values: "b", "kb", "mb", "gb",
     * "tb", "pb". Default is "mb".
     * @param [in] unit the unit.
     */
    void
    SetTargetSizeUnit(const std::string& unit);

    /**
     * @brief Set unit of the target segment size. Supported values: "b", "kb", "mb", "gb",
     * "tb", "pb". Default is "mb".
     * @param [in] unit the unit.
     */
    CompactRequest&
    WithTargetSizeUnit(const std::string& unit);

    /**
     * @brief Get the flag whether it is cluserting compaction or not.
     * @return the clustering compaction.
     */
    bool
    ClusteringCompaction() const;

    /**
     * @brief Set cluserting compaction flag.
     * True: do cluserting compaction, report error if no clustering key.
     * False: do normal compaction.
     * @param [in] clustering_compaction the clustering compaction.
     */
    void
    SetClusteringCompaction(bool clustering_compaction);

    /**
     * @brief Set cluserting compaction flag.
     * True: do cluserting compaction, report error if no clustering key.
     * False: do normal compaction.
     * @param [in] clustering_compaction the clustering compaction.
     */
    CompactRequest&
    WithClusteringCompaction(bool clustering_compaction);

    /**
     * @brief Get the flag whether it is L0 compaction or not.
     * @return true if the compaction target is L0.
     */
    bool
    IsL0() const;

    /**
     * @brief Set L0 compaction flag.
     * True: compact L0 segments only.
     * False: normal compaction.
     * @param [in] is_l0 the is l0.
     */
    void
    SetIsL0(bool is_l0);

    /**
     * @brief Set L0 compaction flag.
     * True: compact L0 segments only.
     * False: normal compaction.
     * @param [in] is_l0 the is l0.
     */
    CompactRequest&
    WithIsL0(bool is_l0);

 private:
    std::string db_name_;
    std::string collection_name_;
    int64_t target_size_{0};
    std::string target_size_unit_{"mb"};
    bool is_clustring_compaction_{false};
    bool is_l0_{false};
};

}  // namespace milvus
