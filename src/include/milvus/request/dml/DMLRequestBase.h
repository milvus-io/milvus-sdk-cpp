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
 * @brief Base class for DML requests.
 */
template <typename T>
class DMLRequestBase {
 protected:
    /**
     * @brief Constructor
     */
    DMLRequestBase() = default;

 public:
    /**
     * @brief Get the target db name
     * @return the database name.
     */
    const std::string&
    DatabaseName() const {
        return db_name_;
    }

    /**
     * @brief Set target db name, use default database if it is empty.
     * @param [in] db_name the DB name.
     */
    void
    SetDatabaseName(const std::string& db_name) {
        db_name_ = db_name;
    }

    /**
     * @brief Set target db name, use default database if it is empty.
     * @param [in] db_name the DB name.
     */
    T&
    WithDatabaseName(const std::string& db_name) {
        SetDatabaseName(db_name);
        return static_cast<T&>(*this);
    }

    /**
     * @brief Get the collection name.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const {
        return collection_name_;
    }

    /**
     * @brief Set the collection name.
     * @param [in] collection_name the collection name.
     */
    void
    SetCollectionName(const std::string& collection_name) {
        collection_name_ = collection_name;
    }

    /**
     * @brief Set name of the collection.
     * @param [in] collection_name the collection name.
     */
    T&
    WithCollectionName(const std::string& collection_name) {
        SetCollectionName(collection_name);
        return static_cast<T&>(*this);
    }

    /**
     * @brief Get the partition name.
     * @return the partition name.
     */
    const std::string&
    PartitionName() const {
        return partition_name_;
    }

    /**
     * @brief Set the partition name.
     * If partition name is empty, will use the default partition.
     * @param [in] partition_name the partition name.
     */
    void
    SetPartitionName(const std::string& partition_name) {
        partition_name_ = partition_name;
    }

    /**
     * @brief Set the partition name.
     * If partition name is empty, will use the default partition.
     * @param [in] partition_name the partition name.
     */
    T&
    WithPartitionName(const std::string& partition_name) {
        SetPartitionName(partition_name);
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
     * On an RLS-enabled collection every row-bearing request must supply a
     * non-blank principal or set SkipRLS(true).
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
     * @brief Get the flag to bypass RLS enforcement for this request.
     * @return the skip flag.
     */
    bool
    SkipRLS() const {
        return skip_rls_;
    }

    /**
     * @brief Set the flag to bypass RLS enforcement for this request.
     * Only allowed when the authenticated user holds the SkipRLS privilege on
     * the collection; `rls.force=true` rejects it.
     * @param [in] skip_rls the skip flag.
     */
    void
    SetSkipRLS(bool skip_rls) {
        skip_rls_ = skip_rls;
    }

    /**
     * @brief Set the flag to bypass RLS enforcement for this request.
     * @param [in] skip_rls the skip flag.
     */
    T&
    WithSkipRLS(bool skip_rls) {
        SetSkipRLS(skip_rls);
        return static_cast<T&>(*this);
    }

 private:
    std::string db_name_;
    std::string collection_name_;
    std::string partition_name_;
    std::string rls_principal_;
    bool skip_rls_{false};
};

}  // namespace milvus
