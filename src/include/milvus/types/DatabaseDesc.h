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
 * @brief Database infomation
 */
class DatabaseDesc {
 public:
    /**
     * @brief Database name
     */
    const std::string&
    Name() const;

    /**
     * @brief Set Database name.
     */
    void
    SetName(std::string name);

    /**
     * @brief Database id.
     */
    int64_t
    ID() const;

    /**
     * @brief Set database id.
     */
    void
    SetID(int64_t id);

    /**
     * @brief Collection alias.
     */
    const std::unordered_map<std::string, std::string>&
    Properties() const;

    /**
     * @brief Set collection alias.
     */
    void
    SetProperties(const std::unordered_map<std::string, std::string>& properties);

    /**
     * @brief Set collection alias.
     */
    void
    SetProperties(std::unordered_map<std::string, std::string>&& properties);

    /**
     * @brief Timestamp when the database created.
     */
    uint64_t
    CreatedTime() const;

    /**
     * @brief Set timestamp when the database created.
     */
    void
    SetCreatedTime(uint64_t ts);

 private:
    std::string db_name_;
    int64_t db_id_;
    std::unordered_map<std::string, std::string> properties_;
    uint64_t created_timestamp_ = 0;
};

}  // namespace milvus
