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

#include "FieldSchema.h"
#include "QueryArguments.h"
#include "SearchArguments.h"

namespace milvus {

/**
 * @brief Base class arguments for MilvusClient::QueryIterator() and SearchIterator().
 */
class IteratorArguments {
 protected:
    IteratorArguments() = default;

 public:
    virtual ~IteratorArguments() = default;

    /**
     * @brief Get the batch size.
     */
    int64_t
    BatchSize() const;

    /**
     * @brief Set the batch size.
     */
    Status
    SetBatchSize(int64_t batch_size);

    /**
     * @brief Get the collection id.
     */
    int64_t
    CollectionID() const;

    /**
     * @brief Set the collection id.
     * No need to manually assign this member, MilvusClient automatically assigns it.
     */
    Status
    SetCollectionID(int64_t id);

    /**
     * @brief Get the primary key field schema.
     */
    const FieldSchema&
    PkSchema() const;

    /**
     * @brief Set the primary key field schema.
     * No need to manually assign this member, MilvusClient automatically assigns it.
     */
    Status
    SetPkSchema(const FieldSchema& schema);

 private:
    int64_t batch_size_{1000};
    int64_t collection_id_{0};
    FieldSchema pk_schema_;
};

/**
 * @brief Arguments for MilvusClient::QueryIterator().
 */
class QueryIteratorArguments : public IteratorArguments, public QueryArguments {
 public:
    /**
     * @brief Get the flag of internal retrieve strategy.
     */
    bool
    ReduceStopForBest() const;

    /**
     * @brief Set the flag of internal retrieve strategy.
     */
    Status
    SetReduceStopForBest(bool reduce_stop_for_best);

 private:
    bool reduce_stop_for_best_{false};
};

/**
 * @brief Arguments for MilvusClient::SearchIterator().
 */
class SearchIteratorArguments : public IteratorArguments, public SearchArguments {};

}  // namespace milvus
