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
     * @brief Get the application-level RLS principal name.
     * @return the principal name.
     */
    const std::string&
    RLSPrincipal() const {
        return rls_principal_;
    }

    /**
     * @brief Set the application-level RLS principal name.
     * On an RLS-enabled collection the import must supply a non-blank principal
     * or set SkipRLS(true); the value is carried in the import options.
     * @param [in] principal the principal name.
     */
    void
    SetRLSPrincipal(const std::string& principal) {
        rls_principal_ = principal;
    }

    /**
     * @brief Set the application-level RLS principal name.
     * @param [in] principal the principal name.
     */
    T&
    WithRLSPrincipal(const std::string& principal) {
        SetRLSPrincipal(principal);
        return static_cast<T&>(*this);
    }

    /**
     * @brief Get the flag to bypass RLS enforcement for this import.
     * @return the skip flag.
     */
    bool
    SkipRLS() const {
        return skip_rls_;
    }

    /**
     * @brief Set the flag to bypass RLS enforcement for this import.
     * Only allowed when the authenticated user holds the SkipRLS privilege on
     * the collection; `rls.force=true` rejects it.
     * @param [in] skip_rls the skip flag.
     */
    void
    SetSkipRLS(bool skip_rls) {
        skip_rls_ = skip_rls;
        skip_rls_set_ = true;
    }

    /**
     * @brief Set the flag to bypass RLS enforcement for this import.
     * @param [in] skip_rls the skip flag.
     */
    T&
    WithSkipRLS(bool skip_rls) {
        SetSkipRLS(skip_rls);
        return static_cast<T&>(*this);
    }

    /**
     * @brief Serialize the request into the REST import job payload.
     * @return the payload.
     */
    virtual nlohmann::json
    ToJson() const {
        nlohmann::json payload = nlohmann::json::object();
        nlohmann::json options = options_;
        if (!rls_principal_.empty()) {
            options["rls_principal"] = rls_principal_;
        }
        if (skip_rls_set_) {
            options["skip_rls"] = skip_rls_ ? "true" : "false";
        }
        if (!options.empty()) {
            payload["options"] = options;
        }
        return payload;
    }

 protected:
    std::string api_key_;
    std::string db_name_;
    std::string collection_name_;
    std::string partition_name_;
    nlohmann::json options_;
    std::string rls_principal_;
    bool skip_rls_{false};
    bool skip_rls_set_{false};
};

}  // namespace milvus
