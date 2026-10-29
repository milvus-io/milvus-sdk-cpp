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
#include "../utils/RpcUtils.h"
#include "../utils/TimeUtils.h"

namespace milvus {

namespace {
const QueryIteratorCursor*
CursorFromRequest(const QueryIteratorArguments&) {
    return nullptr;
}

const QueryIteratorCursor*
CursorFromRequest(const QueryIteratorRequest& args) {
    return args.Cursor();
}
}  // namespace

template class MILVUS_SDK_API Iterator<QueryResults>;

template <typename T>
QueryIteratorImpl<T>::QueryIteratorImpl(const MilvusConnectionPtr& connection, const T& args,
                                        const RetryParam& retry_param, std::string cluster_id) {
    connection_ = connection;
    args_ = args;
    retry_param_ = retry_param;
    cluster_id_ = std::move(cluster_id);
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
            auto status = copyResults(cache_, args_.BatchSize(), cache_.GetRowCount(), new_cache);
            if (!status.IsOk()) {
                return status;
            }
            cache_ = std::move(new_cache);
        } else {
            cache_ = QueryResults();
        }
    } else {
        // perform a query request
        // note that the is_seek flag is set to false, means the REDUCE_STOP_FOR_BEST flag could be true.
        // when REDUCE_STOP_FOR_BEST is true, the milvus server might so some optimize in reduce process
        // and returns results more than batch_size.
        auto filter = setupNextFilter(false);
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
            // the last batch: only the first left_count rows are delivered, so the cursor must
            // be advanced from the trimmed page, not the full batch, otherwise a cursor captured
            // after this page resumes past the last actually-delivered row
            auto status = copyResults(temp_results, 0, left_count, results);
            if (!status.IsOk()) {
                return status;
            }
        }
    }

    // update the cursor from the delivered rows: when a finite Limit truncates the batch, the
    // last delivered row (and its element offset) is what a resumed iterator must continue from
    updateCursor(results);
    returned_count_ += results.GetRowCount();
    return Status::OK();
}

template <typename T>
Status
QueryIteratorImpl<T>::Cursor(QueryIteratorCursor& cursor) const {
    cursor = QueryIteratorCursor();
    auto status = cursor.SetSessionTs(session_ts_);
    if (!status.IsOk()) {
        return status;
    }
    if (next_id_.empty()) {
        // no page has been delivered yet, so there is no primary-key position to resume from;
        // reject the capture rather than emit a pk-less cursor that later fails with a
        // misleading "cursor type does not match the primary key" error
        return {StatusCode::INVALID_ARGUMENT,
                "Query iterator cursor has no primary key; iterate at least one page first"};
    }
    if (args_.PkSchema().FieldDataType() == DataType::VARCHAR) {
        status = cursor.SetStrPk(next_id_);
    } else {
        try {
            status = cursor.SetIntPk(std::stoll(next_id_));
        } catch (const std::exception& e) {
            return {StatusCode::UNKNOWN_ERROR, "Unable to parse iterator primary key, error: " + std::string(e.what())};
        }
    }
    if (!status.IsOk()) {
        return status;
    }
    if (is_element_filter_ && next_element_offset_ >= 0) {
        return cursor.SetLastElementOffset(next_element_offset_);
    }
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

    // detect element_filter iteration: the server caps pages by element count, so exhaustion
    // and resume must be judged by element position rather than entity count
    is_element_filter_ = IsElementFilterExpr(args_.Filter());

    const QueryIteratorCursor* cursor = CursorFromRequest(args_);
    if (cursor != nullptr) {
        // resume from a previously captured cursor: reuse its pk/element position, and skip the
        // offset seek (the cursor already positions past the previously consumed rows).
        if (args_.PkSchema().FieldDataType() == DataType::VARCHAR) {
            if (!cursor->HasStrPk()) {
                return {StatusCode::INVALID_ARGUMENT, "Query iterator cursor type does not match the primary key"};
            }
            next_id_ = cursor->StrPk();
        } else {
            if (!cursor->HasIntPk()) {
                return {StatusCode::INVALID_ARGUMENT, "Query iterator cursor type does not match the primary key"};
            }
            next_id_ = std::to_string(cursor->IntPk());
        }
        if (cursor->HasLastElementOffset()) {
            next_element_offset_ = cursor->LastElementOffset();
        }
        if (cursor->SessionTs() > 0) {
            // pin the new iterator to the same MVCC snapshot and skip the server-side probe.
            session_ts_ = cursor->SessionTs();
            return Status::OK();
        }
        // no captured snapshot timestamp: run the probe to derive the guarantee timestamp.
        QueryResults results;
        auto status = executeQuery(args_.Filter(), 1, false, results);
        if (!status.IsOk()) {
            return status;
        }
        return Status::OK();
    }

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
        auto filter = setupNextFilter(true);
        QueryResults results;
        auto status = executeQuery(filter, batch_size, true, results);
        if (!status.IsOk()) {
            return {status.Code(), "Iterator fails to seek, error: " + status.Message()};
        }

        status = updateCursor(results);
        if (!status.IsOk()) {
            return {status.Code(), "Iterator fails to seek, error: " + status.Message()};
        }

        // The server interprets the offset of an element-filter query in matched-element
        // terms, so a seek page must be counted by elements (sum of element_indices lengths)
        // rather than entity rows; otherwise multi-element entities are under-counted and the
        // seek over-skips, silently dropping rows.
        auto seeked_count = static_cast<int64_t>(is_element_filter_ ? elementCount(results) : results.GetRowCount());
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
// An element-filter cursor resumes at the cursor value from the recorded element offset
// (`pk >= value`); ordinary iterators resume strictly after the value (`pk > value`).
// The strict-exclusive form is used by the offset seek, where Milvus rejects the
// query_iter_last_* element-resume params (they are only accepted when iterator=true).
template <typename T>
std::string
QueryIteratorImpl<T>::setupNextFilter(bool exclusive) {
    if (next_id_.empty()) {
        return args_.Filter();
    }

    std::string pk_name = args_.PkSchema().Name();
    std::string iter_filter;
    const bool inclusive = !exclusive && hasElementCursor();
    if (args_.PkSchema().FieldDataType() == DataType::VARCHAR) {
        iter_filter = pk_name + (inclusive ? " >= " : " > ") + "\"" + EscapeExprStringValue(next_id_) + "\"";
    } else {
        iter_filter = pk_name + (inclusive ? " >= " : " > ") + next_id_;
    }

    std::string user_filter = args_.Filter();
    if (user_filter.empty()) {
        return iter_filter;
    }
    if (is_element_filter_) {
        // Put the PK cursor on the LEFT of AND so that operators with strict position constraints
        // (e.g. element_filter() on struct array, which must be the right-most operand of a logical
        // AND) keep their position across paginated calls.
        return iter_filter + " and ( " + user_filter + " ) ";
    }
    return " ( " + user_filter + " ) " + " and " + iter_filter;
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
        // resume an element-filter page at the element position within the last primary key.
        // the server rejects these params when iterator=false (the offset seek path).
        if (hasElementCursor()) {
            setParamFunc(QUERY_ITER_LAST_PK, next_id_);
            setParamFunc(QUERY_ITER_LAST_ELEMENT_OFFSET, std::to_string(next_element_offset_));
        }
    }

    // reset the limit value since the iterator fetches data batch by batch
    args_.SetLimit(limit);

    auto status =
        ConvertQueryRequest<T>(args_, current_db, rpc_request, cluster_id_, connection_->GetConnectParam().Uri());
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
        target.SetElementIndices(std::vector<std::vector<int64_t>>(src.ElementIndices()));
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

    std::vector<std::vector<int64_t>> result_indices;
    const auto& src_indices = src.ElementIndices();
    if (!src_indices.empty()) {
        // guard against a mismatched element_indices count (e.g. a server returning fewer
        // index lists than rows), mirroring Java's count-vs-row validation
        auto idx_from = from < src_indices.size() ? from : src_indices.size();
        auto idx_to = to < src_indices.size() ? to : src_indices.size();
        if (idx_from < idx_to) {
            result_indices.assign(src_indices.begin() + static_cast<int64_t>(idx_from),
                                  src_indices.begin() + static_cast<int64_t>(idx_to));
        }
    }

    target = QueryResults(std::move(result_fields), args_.OutputFields());
    target.SetElementIndices(std::move(result_indices));
    return Status::OK();
}

// This method update the next_id_ to the last rows of the query result.
// the next_id_ will be used to update the filter expression for the next query.
// For element-filter iterators the server caps pages by element count, so the last entity may
// still have matched elements that were not yet returned; track the element position within the
// last primary key so the resume re-enters the entity (`pk >= value` + query_iter_last_* params).
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

    if (is_element_filter_) {
        const auto& element_indices = results.ElementIndices();
        if (!element_indices.empty()) {
            const auto& last_indices = element_indices.back();
            if (!last_indices.empty()) {
                next_element_offset_ = last_indices.back();
            }
        }
    }
    return Status::OK();
}

template <typename T>
bool
QueryIteratorImpl<T>::hasElementCursor() const {
    return is_element_filter_ && next_element_offset_ >= 0;
}

template <typename T>
uint64_t
QueryIteratorImpl<T>::elementCount(const QueryResults& results) const {
    uint64_t count = 0;
    for (const auto& indices : results.ElementIndices()) {
        count += indices.size();
    }
    // element-level pages always carry element_indices; fall back to entity rows defensively
    return count > 0 ? count : results.GetRowCount();
}

// explicitly instantiation of template methods to avoid link error
template class MILVUS_SDK_API QueryIteratorImpl<QueryIteratorArguments>;
template class MILVUS_SDK_API QueryIteratorImpl<QueryIteratorRequest>;

}  // namespace milvus
