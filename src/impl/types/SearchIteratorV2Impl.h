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

#include <list>
#include <memory>
#include <unordered_map>

#include "../MilvusConnection.h"
#include "milvus.pb.h"
#include "milvus/types/Iterator.h"
#include "milvus/types/IteratorArguments.h"
#include "milvus/types/RetryParam.h"

namespace milvus {

class SearchIteratorV2Impl : public SearchIterator {
 public:
    SearchIteratorV2Impl(MilvusConnectionPtr& connection, const SearchIteratorArguments& args,
                         const RetryParam& retry_param);

    Status
    Next(SingleResult& results) final;

    Status
    Init();

 private:
    Status
    probeForCompability();

    Status
    checkTokenExists(proto::milvus::SearchResults& rpc_response);

    Status
    executeSearch(const SearchIteratorArguments& args, proto::milvus::SearchResults& rpc_response, bool is_probe);

    Status
    next(SingleResultPtr& results);

 private:
    MilvusConnectionPtr connection_;
    SearchIteratorArguments args_;
    RetryParam retry_param_;

    int64_t original_limit_{0};
    int64_t returned_count_{0};
    uint64_t session_ts_{0};
    std::list<SingleResultPtr> cache_;
};

}  // namespace milvus
