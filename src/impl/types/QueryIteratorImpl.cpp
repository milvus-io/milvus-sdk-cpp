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

#include "QueryIteratorImpl.h"

#include <algorithm>

#include "../utils/Constants.h"
#include "../utils/DqlUtils.h"
#include "../utils/GtsDict.h"
#include "../utils/RpcUtils.h"

namespace milvus {
template class Iterator<QueryResults>;

template <typename T>
QueryIteratorImpl<T>::QueryIteratorImpl(const MilvusConnectionPtr& connection, const T& args,
                                        const RetryParam& retry_param) {
    connection_ = connection;
    args_ = args;
    retry_param_ = retry_param;
}

template <typename T>
Status
QueryIteratorImpl<T>::Next(QueryResults& results) {
    results.Clear();

    QueryResults temp_results;
    if (cache_.GetRowCount() >= args_.BatchSize()) {
        // return from cache
        auto status = copyResults(cache_, 0, args_.BatchSize(), temp_results);
        if (!status.IsOk()) {
            return status;
        }

        if (cache_.GetRowCount() >= 2 * args_.BatchSize()) {
            // cut the cache if the cache is big enough for the next batch
            QueryResults new_cache;
            auto status = copyResults(cache_, args_.BatchSize(), cache_.GetRowCount(), temp_results);
            if (!status.IsOk()) {
                return status;
            }
            cache_ = new_cache;
        } else {
            cache_ = QueryResults();
        }
    } else {
        // perform a query request
        // note that the is_seek flag is set to false, means the REDUCE_STOP_FOR_BEST flag could be true.
        // when REDUCE_STOP_FOR_BEST is true, the milvus server might so some optimize in reduce process
        // and returns results more than batch_size.
        auto filter = setupNextFilter();
        QueryResults query_results;
        auto status = executeQuery(filter, args_.BatchSize(), false, query_results);
        if (!status.IsOk()) {
            return status;
        }

        // get one batch result from the query result
        status = copyResults(query_results, 0, args_.BatchSize(), temp_results);
        if (!status.IsOk()) {
            return status;
        }

        // if the query result is big enough for the next batch, cache it
        // for example: batch_size = 5
        //   the executeQuery() migth return 13 rows, we will cache rows from 5 to 10.
        //   if the executeQuery() returns 15 rows, we will cache rows from 5 to 15.
        if (query_results.GetRowCount() >= 2 * args_.BatchSize()) {
            uint64_t cache_to = 2 * args_.BatchSize();
            while (cache_to <= query_results.GetRowCount() - args_.BatchSize()) {
                cache_to += args_.BatchSize();
            }
            status = copyResults(query_results, args_.BatchSize(), cache_to, cache_);
            if (!status.IsOk()) {
                return status;
            }
        }
    }

    if (limit_ < 0) {
        // no limited, continue to fetch
        results = temp_results;
    } else {
        int64_t left_count = limit_ - static_cast<int64_t>(returned_count_);
        if (left_count >= static_cast<int64_t>(temp_results.GetRowCount())) {
            // not enough, continue to fetch
            results = temp_results;
        } else if (left_count > 0) {
            // the last batch
            auto status = copyResults(temp_results, 0, left_count, results);
            if (!status.IsOk()) {
                return status;
            }
        }
    }

    updateCursor(temp_results);
    returned_count_ += temp_results.GetRowCount();
    return Status::OK();
}

template <typename T>
Status
QueryIteratorImpl<T>::Init() {
    // store the limit/offset values, the args's limit/offset will be changed later
    limit_ = args_.Limit();
    offset_ = args_.Offset();

    // reset args_.offset to 0 since the filter expression will be reset to the correct position
    args_.SetOffset(0);

    // run query to setup the session ts
    QueryResults results;
    auto status = executeQuery(args_.Filter(), 1, false, results);
    if (!status.IsOk()) {
        return status;
    }

    // run query to jump offset
    return seek();
}

///////////////////////////////////////////////////////////////////////////////////
// internal methods

// This method is to handle offset
// offset value could be larger than 16384, this method might call query multiple
// times until the "next_id_" is set to the offset position.
template <typename T>
Status
QueryIteratorImpl<T>::seek() {
    if (offset_ == 0) {
        return Status::OK();
    }

    auto offset = offset_;
    while (offset > 0) {
        auto batch_size = std::min(MAX_BATCH_SIZE, offset);
        auto filter = setupNextFilter();
        QueryResults results;
        auto status = executeQuery(filter, batch_size, true, results);
        if (!status.IsOk()) {
            return {status.Code(), "Iterator fails to seek, error: " + status.Message()};
        }

        status = updateCursor(results);
        if (!status.IsOk()) {
            return {status.Code(), "Iterator fails to seek, error: " + status.Message()};
        }

        auto seeked_count = static_cast<int64_t>(results.GetRowCount());
        if (seeked_count == 0) {
            // seek offset has drained all matched results for query iterator, break
            break;
        }
        offset -= seeked_count;
    }
    return Status::OK();
}

// This method return a filter expression for the next query.
// two cases:
//   user inputs expression, "(name != 'xxx') and pk > xx"
//   user doesn't input expression, "pk > xx"
template <typename T>
std::string
QueryIteratorImpl<T>::setupNextFilter() {
    if (next_id_.empty()) {
        return args_.Filter();
    }

    std::string pk_name = args_.PkSchema().Name();
    std::string iter_filter;
    if (args_.PkSchema().FieldDataType() == DataType::VARCHAR) {
        iter_filter = pk_name + " > " + "\"" + next_id_ + "\"";
    } else {
        iter_filter = pk_name + " > " + next_id_;
    }

    std::string user_filter = args_.Filter();
    return user_filter.empty() ? iter_filter : " ( " + user_filter + " ) " + " and " + iter_filter;
}

template <typename T>
Status
QueryIteratorImpl<T>::executeQuery(const std::string& filter, int64_t limit, bool is_seek, QueryResults& results) {
    uint64_t timeout = connection_->GetConnectParam().RpcDeadlineMs();
    std::string current_db =
        args_.DatabaseName().empty() ? connection_->GetConnectParam().DbName() : args_.DatabaseName();
    proto::milvus::QueryRequest rpc_request;
    auto setParamFunc = [&rpc_request](const std::string& key, const std::string& value) {
        auto kv_pair = rpc_request.add_query_params();
        kv_pair->set_key(key);
        kv_pair->set_value(value);
    };
    if (args_.CollectionID() > 0) {
        setParamFunc(COLLECTION_ID, std::to_string(args_.CollectionID()));
    }

    // for seeking process, don't set ITERATOR_FIELD to be true since the server iteration have special reduce logic
    // ReduceStopForBest is for optimize in iteration reduce logic, depends on user input
    if (is_seek) {
        setParamFunc(ITERATOR_FIELD, "False");
        setParamFunc(REDUCE_STOP_FOR_BEST, "False");
    } else {
        setParamFunc(ITERATOR_FIELD, "True");
        setParamFunc(REDUCE_STOP_FOR_BEST, args_.ReduceStopForBest() ? "True" : "False");
    }

    // reset the limit value since the iterator fetches data batch by batch
    args_.SetLimit(limit);

    auto status = ConvertQueryRequest<T>(args_, current_db, rpc_request);
    if (!status.IsOk()) {
        return status;
    }

    // reset filter, the Next() method will change filter every time
    rpc_request.set_expr(filter);

    // for seeking process, no need to return output fields
    if (is_seek) {
        rpc_request.clear_output_fields();
    }

    // the Next() will run into this section
    if (session_ts_ > 0) {
        rpc_request.set_guarantee_timestamp(session_ts_);
    }

    // query rpc call via retry process
    proto::milvus::QueryResults rpc_response;
    auto caller = [&]() { return connection_->Query(rpc_request, rpc_response, GrpcOpts{timeout}); };
    status = Retry(caller, retry_param_);
    if (!status.IsOk()) {
        return status;
    }

    if (session_ts_ == 0) {
        // this section is called at the first time by setupTsByRequest()
        session_ts_ = rpc_response.session_ts();
        if (session_ts_ == 0) {
            // for old milvus versions <= 2.4, the session_ts() might return zero
            // failed to get mvccTs from milvus server, use client-side ts instead
            session_ts_ = static_cast<uint64_t>(MakeMktsFromNowMs());
        }
    }

    return ConvertQueryResults(rpc_response, results);
}

template <typename T>
Status
QueryIteratorImpl<T>::copyResults(const QueryResults& src, uint64_t from, uint64_t to, QueryResults& target) {
    const std::vector<FieldDataPtr>& src_fields = src.OutputFields();
    if ((from == 0 && to == src.GetRowCount()) || src.GetRowCount() == 0) {
        // from begin to end, or the src is empty, no need to copy, return the src
        target = QueryResults(src_fields, args_.OutputFields());
        return Status::OK();
    }

    if (to > src.GetRowCount()) {
        to = src.GetRowCount();
    }

    std::vector<FieldDataPtr> result_fields;
    auto status = CopyFieldsData(src_fields, from, to, result_fields);
    if (!status.IsOk()) {
        return status;
    }

    target = QueryResults(std::move(result_fields), args_.OutputFields());
    return Status::OK();
}

// This method update the next_id_ to the last rows of the query result.
// the next_id_ will be used to update the filter expression for the next query.
template <typename T>
Status
QueryIteratorImpl<T>::updateCursor(const QueryResults& results) {
    if (results.GetRowCount() == 0) {
        // empty result, no need to set the next_id
        return Status::OK();
    }

    std::string pk_name = args_.PkSchema().Name();
    if (args_.PkSchema().FieldDataType() == DataType::VARCHAR) {
        auto pkFieldData = results.OutputField<VarCharFieldData>(pk_name);
        if (pkFieldData == nullptr) {
            return {StatusCode::UNKNOWN_ERROR, "Primary key not found in query results"};
        }
        next_id_ = pkFieldData->Value(pkFieldData->Count() - 1);
    } else {
        auto pkFieldData = results.OutputField<Int64FieldData>(pk_name);
        if (pkFieldData == nullptr) {
            return {StatusCode::UNKNOWN_ERROR, "Primary key not found in query results"};
        }
        auto id = pkFieldData->Value(pkFieldData->Count() - 1);
        next_id_ = std::to_string(id);
    }
    return Status::OK();
}

// explicitly instantiation of template methods to avoid link error
template class QueryIteratorImpl<QueryIteratorArguments>;
template class QueryIteratorImpl<QueryIteratorRequest>;

}  // namespace milvus
