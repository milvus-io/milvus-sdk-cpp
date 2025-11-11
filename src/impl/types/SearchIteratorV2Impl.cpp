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

#include "SearchIteratorV2Impl.h"

#include <stdexcept>

#include "../utils/CompareUtils.h"
#include "../utils/Constants.h"
#include "../utils/DqlUtils.h"
#include "../utils/GtsDict.h"
#include "../utils/RpcUtils.h"
#include "../utils/TypeUtils.h"
#include "SearchIteratorImpl.h"

namespace milvus {
template class Iterator<SingleResult>;

SearchIteratorV2Impl::SearchIteratorV2Impl(MilvusConnectionPtr& connection, const SearchIteratorArguments& args,
                                           const RetryParam& retry_param) {
    connection_ = connection;
    args_ = args;
    original_limit_ = args.Limit();
    retry_param_ = retry_param;
}

Status
SearchIteratorV2Impl::Next(SingleResult& results) {
    results.Clear();

    // returned count already meet the limit value
    if (original_limit_ > 0 && returned_count_ >= original_limit_) {
        return Status::OK();
    }

    auto target_len = static_cast<int64_t>(args_.BatchSize());
    // the last batch might returns a few items since the limit value is almost meet
    if (original_limit_ > 0) {
        auto left_count = original_limit_ - returned_count_;
        target_len = target_len > left_count ? left_count : target_len;
    }

    while (true) {
        SingleResultPtr single_result;
        auto status = next(single_result);
        if (!status.IsOk()) {
            return status;
        }
        auto result_count = single_result->GetRowCount();
        if (result_count == 0) {
            break;
        }

        cache_.emplace_back(std::move(single_result));
        auto cache_count = SearchIteratorImpl::CachedCount(cache_);
        if (cache_count >= static_cast<uint64_t>(target_len)) {
            break;
        }
    }

    // return batch from the cache if cache is big enough
    auto status = SearchIteratorImpl::FetchPageFromCache(cache_, args_, target_len, results);
    if (!status.IsOk()) {
        return status;
    }
    returned_count_ += static_cast<int64_t>(results.GetRowCount());

    return Status::OK();
}

Status
SearchIteratorV2Impl::Init() {
    auto status = SearchIteratorImpl::CheckInput(args_);
    if (!status.IsOk()) {
        return status;
    }

    args_.SetLimit(static_cast<int64_t>(args_.BatchSize()));
    args_.AddExtraParam(COLLECTION_ID, std::to_string(args_.CollectionID()));
    args_.AddExtraParam(ITERATOR_FIELD, "True");
    args_.AddExtraParam(ITER_SEARCH_V2_KEY, "True");
    args_.AddExtraParam(ITER_SEARCH_BATCH_SIZE_KEY, std::to_string(args_.BatchSize()));

    status = probeForCompability();
    if (!status.IsOk()) {
        return status;
    }

    return Status::OK();
}

///////////////////////////////////////////////////////////////////////////////////
// internal methods
Status
SearchIteratorV2Impl::probeForCompability() {
    SearchIteratorArguments temp_args = args_;
    temp_args.SetLimit(1);
    temp_args.AddExtraParam(ITER_SEARCH_BATCH_SIZE_KEY, "1");
    proto::milvus::SearchResults rpc_response;
    auto status = executeSearch(temp_args, rpc_response, true);
    if (!status.IsOk()) {
        return {status.Code(), "Fail to init search iterator, error: " + status.Message()};
    }

    return checkTokenExists(rpc_response);
}

Status
SearchIteratorV2Impl::checkTokenExists(proto::milvus::SearchResults& rpc_response) {
    proto::schema::SearchResultData data = rpc_response.results();
    auto token = data.search_iterator_v2_results().token();
    if (token.empty()) {
        std::string msg =
            "The server does not support Search Iterator V2. The search_iterator (v1) is used instead. Please upgrade "
            "your Milvus server version to 2.5.2 and later, or use a pymilvus version before 2.5.3 (excluded) to avoid "
            "this issue.";
        return {StatusCode::NOT_SUPPORTED, msg};
    }

    return Status::OK();
}

Status
SearchIteratorV2Impl::executeSearch(const SearchIteratorArguments& args, proto::milvus::SearchResults& rpc_response,
                                    bool is_probe) {
    uint64_t timeout = connection_->GetConnectParam().RpcDeadlineMs();
    std::string current_db =
        args.DatabaseName().empty() ? connection_->GetConnectParam().DbName() : args.DatabaseName();

    proto::milvus::SearchRequest rpc_request;
    auto status = ConvertSearchRequest(args, current_db, rpc_request);
    if (!status.IsOk()) {
        return status;
    }

    if (is_probe || session_ts_ == 0) {
        // probe method and the first time search no need to set guarantee_timestamp
        rpc_request.set_guarantee_timestamp(0);
    } else {
        // session_ts_ value is set at the first time to search
        // from the second time search, guarantee_timestamp is assigned by session_ts_
        rpc_request.set_guarantee_timestamp(session_ts_);
    }

    // query rpc call via retry process
    auto caller = [&]() { return connection_->Search(rpc_request, rpc_response, GrpcOpts{timeout}); };
    status = Retry(caller, retry_param_);
    if (!status.IsOk()) {
        return status;
    }

    if (!is_probe && session_ts_ == 0) {
        // for old milvus versions < 2.5.0, the SearchResults has no session_ts
        // use client-side ts instead, else use the ts returned by SearchResults
        auto ts = rpc_response.session_ts();
        session_ts_ = (ts == 0) ? static_cast<uint64_t>(MakeMktsFromNowMs()) : ts;
    }

    return Status::OK();
}

Status
SearchIteratorV2Impl::next(SingleResultPtr& results) {
    proto::milvus::SearchResults rpc_response;
    auto status = executeSearch(args_, rpc_response, false);
    if (!status.IsOk()) {
        return status;
    }

    status = checkTokenExists(rpc_response);
    if (!status.IsOk()) {
        return status;
    }

    // set the bound for the next search, the bound must be a string of double precise
    // you will get bug if you treat it as float
    auto rpc_results = rpc_response.results().search_iterator_v2_results();
    auto bound = doubleToString(static_cast<double>(rpc_results.last_bound()));
    args_.AddExtraParam(ITER_SEARCH_LAST_BOUND_KEY, bound);

    const auto& params = args_.ExtraParams();
    if (params.find(ITER_SEARCH_ID_KEY) == params.end()) {
        args_.AddExtraParam(ITER_SEARCH_ID_KEY, rpc_results.token());
    }

    SearchResults search_results;
    status = ConvertSearchResults(rpc_response, args_.PkSchema().Name(), search_results);
    if (!status.IsOk()) {
        return status;
    }

    // nq = 1, the search_results must contains a SingleResult. Otherwise it is a server-side bug.
    if (search_results.Results().size() != 1) {
        return {StatusCode::UNKNOWN_ERROR, "the server returns an unexpected search result"};
    }

    auto& single_result = search_results.Results().at(0);
    results = std::make_shared<SingleResult>(single_result);
    return Status::OK();
}

}  // namespace milvus
