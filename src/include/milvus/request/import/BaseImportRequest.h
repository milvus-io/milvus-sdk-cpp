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

#include "milvus/thirdparty/nlohmann/json.hpp"

namespace milvus {

/**
 * @brief Base class for data import requests.
 */
template <typename T>
class BaseImportRequest {
 protected:
    /**
     * @brief Constructor
     */
    BaseImportRequest() = default;

 public:
    /**
     * @brief Destructor
     */
    virtual ~BaseImportRequest() = default;

    /**
     * @brief Get the API key used for the cloud API, or the userName:password for Milvus directly.
     * @return the API key.
     */
    const std::string&
    ApiKey() const {
        return api_key_;
    }

    /**
     * @brief Set the API key.
     * @param [in] api_key the API key.
     */
    void
    SetApiKey(const std::string& api_key) {
        api_key_ = api_key;
    }

    /**
     * @brief Set the API key.
     * @param [in] api_key the API key.
     */
    T&
    WithApiKey(const std::string& api_key) {
        SetApiKey(api_key);
        return static_cast<T&>(*this);
    }

    /**
     * @brief Get the target database name.
     * Note: create/list calls deliver the database in the JSON body, while describe/commit/abort
     * calls send it via the DB-Name header.
     * @return the database name.
     */
    const std::string&
    DatabaseName() const {
        return db_name_;
    }

    /**
     * @brief Set the target database name.
     * @param [in] db_name the database name.
     */
    void
    SetDatabaseName(const std::string& db_name) {
        db_name_ = db_name;
    }

    /**
     * @brief Set the target database name.
     * @param [in] db_name the database name.
     */
    T&
    WithDatabaseName(const std::string& db_name) {
        SetDatabaseName(db_name);
        return static_cast<T&>(*this);
    }

    /**
     * @brief Get the target collection name.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const {
        return collection_name_;
    }

    /**
     * @brief Set the target collection name.
     * @param [in] collection_name the collection name.
     */
    void
    SetCollectionName(const std::string& collection_name) {
        collection_name_ = collection_name;
    }

    /**
     * @brief Set the target collection name.
     * @param [in] collection_name the collection name.
     */
    T&
    WithCollectionName(const std::string& collection_name) {
        SetCollectionName(collection_name);
        return static_cast<T&>(*this);
    }

    /**
     * @brief Get the target partition name.
     * @return the partition name.
     */
    const std::string&
    PartitionName() const {
        return partition_name_;
    }

    /**
     * @brief Set the target partition name.
     * @param [in] partition_name the partition name.
     */
    void
    SetPartitionName(const std::string& partition_name) {
        partition_name_ = partition_name;
    }

    /**
     * @brief Set the target partition name.
     * @param [in] partition_name the partition name.
     */
    T&
    WithPartitionName(const std::string& partition_name) {
        SetPartitionName(partition_name);
        return static_cast<T&>(*this);
    }

    /**
     * @brief Get the additional import options.
     * @return the options.
     */
    const nlohmann::json&
    Options() const {
        return options_;
    }

    /**
     * @brief Set the additional import options.
     * @param [in] options the options.
     */
    void
    SetOptions(nlohmann::json&& options) {
        options_ = std::move(options);
    }

    /**
     * @brief Set the additional import options.
     * @param [in] options the options.
     */
    T&
    WithOptions(nlohmann::json&& options) {
        SetOptions(std::move(options));
        return static_cast<T&>(*this);
    }

    /**
     * @brief Serialize the request into the REST import job payload.
     * @return the payload.
     */
    virtual nlohmann::json
    ToJson() const {
        nlohmann::json payload = nlohmann::json::object();
        if (!options_.empty()) {
            payload["options"] = options_;
        }
        return payload;
    }

 protected:
    std::string api_key_;
    std::string db_name_;
    std::string collection_name_;
    std::string partition_name_;
    nlohmann::json options_;
};

}  // namespace milvus
