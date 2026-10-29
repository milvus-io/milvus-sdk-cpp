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
#include <memory>
#include <string>
#include <vector>

#include "QueryResults.h"
#include "SearchResults.h"
#include "milvus/Export.h"
#include "milvus/Status.h"

namespace milvus {

/**
 * @brief A serializable snapshot of a QueryIterator position, used to resume pagination.
 *
 * Captures the session timestamp of the original iterator, the last primary key returned,
 * and (for element-filter iterators) the last matched element offset. Pass it back through
 * MilvusClientV2::QueryIterator() request's WithCursor() to resume from exactly this position.
 */
class MILVUS_SDK_API QueryIteratorCursor {
 public:
    /**
     * @brief Constructor
     */
    QueryIteratorCursor();

    /**
     * @brief Get the MVCC session timestamp pinned from the original iterator.
     * @return the session timestamp.
     */
    uint64_t
    SessionTs() const;

    /**
     * @brief Set the MVCC session timestamp.
     * @param [in] session_ts the session timestamp.
     */
    Status
    SetSessionTs(uint64_t session_ts);

    /**
     * @brief Set the MVCC session timestamp.
     * @param [in] session_ts the session timestamp.
     */
    QueryIteratorCursor&
    WithSessionTs(uint64_t session_ts);

    /**
     * @brief Check whether an Int64 primary-key position is present.
     * @return true if the Int64 primary key is set.
     */
    bool
    HasIntPk() const;

    /**
     * @brief Get the Int64 primary-key position.
     * @return the Int64 primary key.
     */
    int64_t
    IntPk() const;

    /**
     * @brief Set the Int64 primary-key position.
     * @param [in] pk the Int64 primary key.
     */
    Status
    SetIntPk(int64_t pk);

    /**
     * @brief Set the Int64 primary-key position.
     * @param [in] pk the Int64 primary key.
     */
    QueryIteratorCursor&
    WithIntPk(int64_t pk);

    /**
     * @brief Check whether a VarChar primary-key position is present.
     * @return true if the VarChar primary key is set.
     */
    bool
    HasStrPk() const;

    /**
     * @brief Get the VarChar primary-key position.
     * @return the VarChar primary key.
     */
    const std::string&
    StrPk() const;

    /**
     * @brief Set the VarChar primary-key position.
     * @param [in] pk the VarChar primary key.
     */
    Status
    SetStrPk(const std::string& pk);

    /**
     * @brief Set the VarChar primary-key position.
     * @param [in] pk the VarChar primary key.
     */
    QueryIteratorCursor&
    WithStrPk(const std::string& pk);

    /**
     * @brief Check whether the element position within the last primary key is present.
     * @return true if the element offset is set.
     */
    bool
    HasLastElementOffset() const;

    /**
     * @brief Get the element position within the last primary key.
     * Only meaningful for element-filter iterators; ordinary iterators leave it unset.
     * @return the last element offset.
     */
    int64_t
    LastElementOffset() const;

    /**
     * @brief Set the element position within the last primary key.
     * @param [in] offset the last element offset.
     */
    Status
    SetLastElementOffset(int64_t offset);

    /**
     * @brief Set the element position within the last primary key.
     * @param [in] offset the last element offset.
     */
    QueryIteratorCursor&
    WithLastElementOffset(int64_t offset);

 private:
    uint64_t session_ts_{0};
    bool has_int_pk_{false};
    int64_t int_pk_{0};
    bool has_str_pk_{false};
    std::string str_pk_;
    bool has_last_element_offset_{false};
    int64_t last_element_offset_{0};
};

/**
 * @brief Base class for MilvusClient::QueryIterator() and SearchIterator().
 */
template <typename T>
class Iterator {
 protected:
    /**
     * @brief Constructor
     */
    Iterator() = default;

 public:
    virtual ~Iterator() = default;

    /**
     * @brief Get next batch of results.
     * Note: this method is not designed to be called in multi-thread, it is not thread-safe.
     *
     * @return QueryResults or SingleResult
     * @param [in] results the results.
     */
    virtual Status
    Next(T& results) = 0;

    /**
     * @brief Capture the current position as a resumable cursor.
     * Only QueryIterator supports this; SearchIterator returns StatusCode::NOT_SUPPORTED.
     * @param [in] cursor the cursor to fill.
     * @return the status.
     */
    virtual Status
    Cursor(QueryIteratorCursor& cursor) const {
        return {StatusCode::NOT_SUPPORTED, "cursor is only supported for QueryIterator"};
    }
};

extern template class MILVUS_SDK_API Iterator<QueryResults>;
extern template class MILVUS_SDK_API Iterator<SingleResult>;

using QueryIterator = Iterator<QueryResults>;
using SearchIterator = Iterator<SingleResult>;

using QueryIteratorPtr = std::shared_ptr<QueryIterator>;
using SearchIteratorPtr = std::shared_ptr<SearchIterator>;

}  // namespace milvus
