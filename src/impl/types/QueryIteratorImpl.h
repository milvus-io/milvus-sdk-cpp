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

#include "../MilvusConnection.h"
#include "milvus/types/FieldSchema.h"
#include "milvus/types/Iterator.h"
#include "milvus/types/IteratorArguments.h"
#include "milvus/types/RetryParam.h"

namespace milvus {
class QueryIteratorImpl : public QueryIterator {
 public:
    QueryIteratorImpl(MilvusConnectionPtr& connection, const QueryIteratorArguments& args,
                      const RetryParam& retry_param);

    Status
    Next(QueryResults& results) final;

 private:
    void
    init();

    void
    seek();

    std::string
    setupNextFilter();

    Status
    executeQuery(const std::string& filter, int64_t limit, bool is_seek, QueryResults& results);

    Status
    copyResults(const QueryResults& src, uint64_t from, uint64_t to, QueryResults& target);

    Status
    updateCursor(const QueryResults& results);

 private:
    MilvusConnectionPtr connection_;
    QueryIteratorArguments args_;
    RetryParam retry_param_;

    int64_t offset_{0};
    int64_t limit_{0};

    uint64_t session_ts_{0};
    std::string next_id_;
    uint64_t returned_count_{0};

    QueryResults cache_;
};

}  // namespace milvus
