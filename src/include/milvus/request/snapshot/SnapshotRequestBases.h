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

namespace milvus {

/**
 * @brief CRTP base class of snapshot request types that target a database and collection.
 *
 * @tparam T derived request type returned by the fluent With* methods.
 */
template <typename T>
class SnapshotRequestBase {
 protected:
    /**
     * @brief Constructor
     */
    SnapshotRequestBase() = default;

 public:
    /**
     * @brief Get the target database name.
     * @return database name.
     */
    const std::string&
    DatabaseName() const {
        return db_name_;
    }

    /**
     * @brief Set the target database name.
     * @param [in] db_name database name.
     */
    void
    SetDatabaseName(const std::string& db_name) {
        db_name_ = db_name;
    }

    /**
     * @brief Set the target database name.
     * @param [in] db_name database name.
     */
    T&
    WithDatabaseName(const std::string& db_name) {
        SetDatabaseName(db_name);
        return static_cast<T&>(*this);
    }

    /**
     * @brief Get the target collection name.
     * @return collection name.
     */
    const std::string&
    CollectionName() const {
        return collection_name_;
    }

    /**
     * @brief Set the target collection name.
     * @param [in] collection_name collection name.
     */
    void
    SetCollectionName(const std::string& collection_name) {
        collection_name_ = collection_name;
    }

    /**
     * @brief Set the target collection name.
     * @param [in] collection_name collection name.
     */
    T&
    WithCollectionName(const std::string& collection_name) {
        SetCollectionName(collection_name);
        return static_cast<T&>(*this);
    }

 protected:
    std::string db_name_;
    std::string collection_name_;
};

/**
 * @brief CRTP base class of snapshot request types that also target a snapshot name.
 *
 * @tparam T derived request type returned by the fluent With* methods.
 */
template <typename T>
class SnapshotNameRequestBase : public SnapshotRequestBase<T> {
 protected:
    /**
     * @brief Constructor
     */
    SnapshotNameRequestBase() = default;

 public:
    /**
     * @brief Get the target snapshot name.
     * @return snapshot name.
     */
    const std::string&
    SnapshotName() const {
        return snapshot_name_;
    }

    /**
     * @brief Set the target snapshot name.
     * @param [in] snapshot_name snapshot name.
     */
    void
    SetSnapshotName(const std::string& snapshot_name) {
        snapshot_name_ = snapshot_name;
    }

    /**
     * @brief Set the target snapshot name.
     * @param [in] snapshot_name snapshot name.
     */
    T&
    WithSnapshotName(const std::string& snapshot_name) {
        SetSnapshotName(snapshot_name);
        return static_cast<T&>(*this);
    }

 protected:
    std::string snapshot_name_;
};

}  // namespace milvus
