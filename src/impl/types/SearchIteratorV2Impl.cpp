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

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <milvus/thirdparty/nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

#include "../utils/CompareUtils.h"
#include "../utils/Constants.h"
#include "../utils/DqlUtils.h"
#include "../utils/ExtraParamUtils.h"
#include "../utils/RpcUtils.h"
#include "../utils/TimeUtils.h"
#include "../utils/TypeUtils.h"
#include "SearchIteratorImpl.h"

namespace milvus {

namespace {
const char* kCursorVersion = "search_iter_cursor_version";
const char* kLastPkType = "search_iter_last_pk_type";
const char* kLastPk = "search_iter_last_pk";

void
RemoveRpcParam(proto::milvus::SearchRequest& request, const std::string& key) {
    auto* params = request.mutable_search_params();
    for (int i = params->size() - 1; i >= 0; --i) {
        if (params->Get(i).key() == key) {
            params->DeleteSubrange(i, 1);
        }
    }
}

void
SetRpcParam(proto::milvus::SearchRequest& request, const std::string& key, const std::string& value) {
    RemoveRpcParam(request, key);
    auto* pair = request.add_search_params();
    pair->set_key(key);
    pair->set_value(value);
}

std::string
RpcParam(const proto::milvus::SearchRequest& request, const std::string& key) {
    for (const auto& pair : request.search_params()) {
        if (pair.key() == key) {
            return pair.value();
        }
    }
    return "";
}

std::string
ExtraInfo(const proto::milvus::SearchResults& response, const std::string& key) {
    const auto& extra = response.status().extra_info();
    auto found = extra.find(key);
    return found == extra.end() ? "" : found->second;
}

std::string
BoundText(float value) {
    std::ostringstream text;
    text.imbue(std::locale::classic());
    text << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return text.str();
}
}  // namespace

template <typename T>
SearchIteratorV2Impl<T>::SearchIteratorV2Impl(const MilvusConnectionPtr& connection, const T& args,
                                              const RetryParam& retry_param, std::string cluster_id) {
    connection_ = connection;
    args_ = args;
    if (args_.DatabaseName().empty()) {
        const auto& db = connection_->GetConnectParam().DbName();
        args_.SetDatabaseName(db.empty() ? "default" : db);
    }
    original_limit_ = args.Limit();
    retry_param_ = retry_param;
    cluster_id_ = std::move(cluster_id);
}

template <typename T>
Status
SearchIteratorV2Impl<T>::Next(SingleResult& results) {
    results.Clear();
    try {
        if (original_limit_ == 0 || (original_limit_ > 0 && returned_count_ >= original_limit_)) {
            return Status::OK();
        }
        auto target = static_cast<int64_t>(args_.BatchSize());
        if (original_limit_ > 0) {
            target = std::min(target, original_limit_ - returned_count_);
        }
        while (!finished_ && SearchIteratorImpl<T>::CachedCount(cache_) < static_cast<uint64_t>(target)) {
            SingleResultPtr raw_page;
            auto status = loadPending(raw_page);
            if (!status.IsOk()) {
                return status;
            }
            const bool exhausted = raw_page->GetRowCount() == 0;
            auto filtered = raw_page;
            if (raw_page->GetRowCount() > 0 && args_.ExternalFilterFunc()) {
                status = args_.ExternalFilterFunc()(*filtered);
                if (!status.IsOk()) {
                    return status;
                }
            }
            // A failed page filter leaves the raw response and cursor pending.
            // preparePending decodes a fresh copy on the next call.
            std::unordered_set<std::string> added_pks;
            if (pending_mode_ == CursorMode::PRIMARY_KEY && filtered->GetRowCount() > 0) {
                const auto ids = filtered->Ids();
                std::vector<uint64_t> keep;
                keep.reserve(filtered->GetRowCount());
                const auto& integer_ids = ids.IntIDArray();
                const auto& string_ids = ids.StrIDArray();
                for (uint64_t i = 0; i < filtered->GetRowCount(); ++i) {
                    const auto key = integer_ids.empty() ? string_ids.at(i) : std::to_string(integer_ids.at(i));
                    if (accepted_pks_.count(key) == 0 && added_pks.insert(key).second) {
                        keep.emplace_back(i);
                    }
                }
                status = filtered->FilterRows(keep);
                if (!status.IsOk()) {
                    return status;
                }
            }
            auto next_cache = cache_;
            if (filtered->GetRowCount() > 0) {
                next_cache.emplace_back(std::move(filtered));
            }
            auto next_request = pending_request_;
            try {
                for (const auto& key : added_pks) {
                    accepted_pks_.insert(key);
                }
            } catch (...) {
                for (const auto& key : added_pks) {
                    accepted_pks_.erase(key);
                }
                throw;
            }
            request_.Swap(&next_request);
            mode_ = pending_mode_;
            has_pending_ = false;
            finished_ = exhausted;
            cache_ = std::move(next_cache);
        }
        auto next_cache = cache_;
        SingleResult page;
        auto status = SearchIteratorImpl<T>::FetchPageFromCache(next_cache, args_.OutputFields(), target, page);
        if (!status.IsOk()) {
            return status;
        }
        returned_count_ += static_cast<int64_t>(page.GetRowCount());
        cache_ = std::move(next_cache);
        results = page;
        return Status::OK();
    } catch (const std::exception& error) {
        return {StatusCode::UNKNOWN_ERROR, std::string("search iterator page failed: ") + error.what()};
    } catch (...) {
        return {StatusCode::UNKNOWN_ERROR, "search iterator page filter threw an unknown exception"};
    }
}

template <typename T>
Status
SearchIteratorV2Impl<T>::Init() {
    try {
        auto status = SearchIteratorImpl<T>::CheckInput(args_.TargetVectors(), args_.ExtraParams(), args_.BatchSize(),
                                                        args_.MetricType());
        if (!status.IsOk()) {
            return status;
        }
        if (original_limit_ == 0) {
            finished_ = true;
            return Status::OK();
        }
        args_.SetLimit(static_cast<int64_t>(args_.BatchSize()));
        args_.AddExtraParam(COLLECTION_ID, std::to_string(args_.CollectionID()));
        args_.AddExtraParam(ITERATOR_FIELD, "True");
        args_.AddExtraParam(ITER_SEARCH_V2_KEY, "True");
        args_.AddExtraParam(ITER_SEARCH_BATCH_SIZE_KEY, std::to_string(args_.BatchSize()));
        status = ConvertSearchRequest<T>(args_, args_.DatabaseName(), request_, cluster_id_,
                                         connection_->GetConnectParam().Uri());
        if (!status.IsOk()) {
            return status;
        }
        const auto requested_version = RpcParam(request_, kCursorVersion);
        if (!requested_version.empty() && requested_version != "2") {
            return {StatusCode::INVALID_ARGUMENT, "unsupported search iterator cursor version: " + requested_version};
        }
        if (requested_version == "2" && (!RpcParam(request_, ITER_SEARCH_ID_KEY).empty() ||
                                         !RpcParam(request_, ITER_SEARCH_LAST_BOUND_KEY).empty())) {
            return {StatusCode::INVALID_ARGUMENT,
                    "PK cursor mode requires a complete typed cursor; legacy token/bound continuation must omit cursor "
                    "version 2"};
        }
        // These negotiated controls are outer RPC parameters, never ANN index JSON.
        for (auto& pair : *request_.mutable_search_params()) {
            if (pair.key() == PARAMS) {
                auto nested = nlohmann::json::parse(pair.value());
                nested.erase(kCursorVersion);
                nested.erase(kLastPkType);
                nested.erase(kLastPk);
                pair.set_value(nested.dump());
            }
        }
        RemoveRpcParam(request_, kLastPkType);
        RemoveRpcParam(request_, kLastPk);
        if (requested_version != "2" || !RpcParam(request_, ITER_SEARCH_ID_KEY).empty() ||
            !RpcParam(request_, ITER_SEARCH_LAST_BOUND_KEY).empty()) {
            mode_ = CursorMode::DISTANCE;
            RemoveRpcParam(request_, kCursorVersion);
        } else {
            SetRpcParam(request_, kCursorVersion, "2");
        }
        request_.set_guarantee_timestamp(0);
        status = executeSearch(pending_response_);
        if (!status.IsOk()) {
            return status;
        }
        SingleResultPtr page;
        status = preparePending(page);
        if (!status.IsOk()) {
            return status;
        }
        has_pending_ = true;
        initialized_ = true;
        return Status::OK();
    } catch (const std::exception& error) {
        return {StatusCode::UNKNOWN_ERROR, std::string("search iterator initialization failed: ") + error.what()};
    }
}

template <typename T>
Status
SearchIteratorV2Impl<T>::executeSearch(proto::milvus::SearchResults& response) {
    auto timeout = connection_->GetConnectParam().RpcDeadlineMs();
    auto caller = [&]() { return connection_->Search(request_, response, GrpcOpts{timeout}); };
    return Retry(caller, retry_param_);
}

template <typename T>
Status
SearchIteratorV2Impl<T>::loadPending(SingleResultPtr& results) {
    if (!has_pending_) {
        pending_response_.Clear();
        auto status = executeSearch(pending_response_);
        if (!status.IsOk()) {
            return status;
        }
    }
    auto status = preparePending(results);
    has_pending_ = status.IsOk();
    return status;
}

template <typename T>
Status
SearchIteratorV2Impl<T>::preparePending(SingleResultPtr& results) {
    const auto& data = pending_response_.results();
    const auto& info = data.search_iterator_v2_results();
    const auto version = ExtraInfo(pending_response_, kCursorVersion);
    if (!data.has_search_iterator_v2_results() || info.token().empty()) {
        if (!initialized_ && version.empty()) {
            return {StatusCode::NOT_SUPPORTED, "server does not support Search Iterator V2"};
        }
        return {StatusCode::UNKNOWN_ERROR, "search iterator cursor response has no V2 metadata"};
    }
    if (!version.empty() && version != "2") {
        return {StatusCode::UNKNOWN_ERROR, "unsupported search iterator cursor version: " + version};
    }
    if (version == "2" && RpcParam(request_, kCursorVersion) != "2") {
        return {StatusCode::UNKNOWN_ERROR, "server activated PK cursor mode without client opt-in"};
    }
    const auto next_mode = version.empty() ? CursorMode::DISTANCE : CursorMode::PRIMARY_KEY;
    if (mode_ != CursorMode::UNINITIALIZED && mode_ != next_mode) {
        return {StatusCode::UNKNOWN_ERROR, "search iterator cursor mode changed between pages"};
    }
    const auto token = RpcParam(request_, ITER_SEARCH_ID_KEY);
    if (!token.empty() && token != info.token()) {
        return {StatusCode::UNKNOWN_ERROR, "search iterator token changed between pages"};
    }
    int64_t count = 0;
    std::string pk_type;
    std::string last_pk;
    if (next_mode == CursorMode::PRIMARY_KEY) {
        if (data.num_queries() != 1 || data.topks_size() != 1 || data.topks(0) < 0 ||
            static_cast<uint64_t>(data.topks(0)) > args_.BatchSize() || data.scores_size() != data.topks(0) ||
            !std::isfinite(info.last_bound())) {
            return {StatusCode::UNKNOWN_ERROR, "invalid search iterator result shape or bound"};
        }
        for (const auto score : data.scores()) {
            if (!std::isfinite(score)) {
                return {StatusCode::UNKNOWN_ERROR, "non-finite search iterator score"};
            }
        }
        count = data.topks(0);
        if (args_.PkSchema().FieldDataType() == DataType::INT64) {
            pk_type = "int64";
            if (data.ids().int_id().data_size() != count || (count > 0 && !data.ids().has_int_id())) {
                return {StatusCode::UNKNOWN_ERROR, "search iterator int64 IDs do not match result count"};
            }
            if (count > 0) {
                last_pk = std::to_string(data.ids().int_id().data(static_cast<int>(count - 1)));
            }
        } else if (args_.PkSchema().FieldDataType() == DataType::VARCHAR) {
            pk_type = "varchar";
            if (data.ids().str_id().data_size() != count || (count > 0 && !data.ids().has_str_id())) {
                return {StatusCode::UNKNOWN_ERROR, "search iterator varchar IDs do not match result count"};
            }
            if (count > 0) {
                last_pk = data.ids().str_id().data(static_cast<int>(count - 1));
            }
        } else {
            return {StatusCode::UNKNOWN_ERROR, "unsupported search iterator primary key schema"};
        }
    }
    auto candidate = request_;
    if (next_mode == CursorMode::PRIMARY_KEY) {
        if (count > 0) {
            const auto& extra = pending_response_.status().extra_info();
            if (ExtraInfo(pending_response_, kLastPkType) != pk_type || extra.find(kLastPk) == extra.end() ||
                ExtraInfo(pending_response_, kLastPk) != last_pk ||
                info.last_bound() != data.scores(static_cast<int>(count - 1))) {
                return {StatusCode::UNKNOWN_ERROR, "search iterator cursor does not match last result"};
            }
            SetRpcParam(candidate, kLastPkType, pk_type);
            SetRpcParam(candidate, kLastPk, last_pk);
        }
        SetRpcParam(candidate, kCursorVersion, "2");
    } else {
        RemoveRpcParam(candidate, kCursorVersion);
        RemoveRpcParam(candidate, kLastPkType);
        RemoveRpcParam(candidate, kLastPk);
    }
    if (candidate.guarantee_timestamp() == 0) {
        if (next_mode == CursorMode::PRIMARY_KEY && pending_response_.session_ts() == 0) {
            return {StatusCode::UNKNOWN_ERROR, "search iterator PK cursor response has no snapshot timestamp"};
        }
        candidate.set_guarantee_timestamp(pending_response_.session_ts());
    }
    SetRpcParam(candidate, ITER_SEARCH_ID_KEY, info.token());
    SetRpcParam(candidate, ITER_SEARCH_LAST_BOUND_KEY, BoundText(info.last_bound()));
    SearchResults decoded;
    auto status = ConvertSearchResults(pending_response_, args_.PkSchema().Name(), decoded);
    if (!status.IsOk()) {
        return status;
    }
    if (decoded.Results().size() != 1 ||
        (next_mode == CursorMode::PRIMARY_KEY && decoded.Results().at(0).GetRowCount() != static_cast<size_t>(count))) {
        return {StatusCode::UNKNOWN_ERROR, "search iterator decoded result count is invalid"};
    }
    results = std::make_shared<SingleResult>(decoded.Results().at(0));
    pending_request_ = std::move(candidate);
    pending_mode_ = next_mode;
    return Status::OK();
}

// explicitly instantiation of template methods to avoid link error
template class MILVUS_SDK_API SearchIteratorV2Impl<SearchIteratorArguments>;
template class MILVUS_SDK_API SearchIteratorV2Impl<SearchIteratorRequest>;

}  // namespace milvus
