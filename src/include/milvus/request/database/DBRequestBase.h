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
 * @brief Base class for database requests, except the ListDatabasesRequest.
 */
template <typename T>
class DBRequestBase {
 protected:
    /**
     * @brief Constructor
     */
    DBRequestBase() = default;

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

 private:
    std::string db_name_;
};

}  // namespace milvus
