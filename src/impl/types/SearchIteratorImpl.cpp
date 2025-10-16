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

#include "SearchIteratorImpl.h"

#include <stdexcept>

#include "../utils/CompareUtils.h"
#include "../utils/Constants.h"
#include "../utils/DqlUtils.h"
#include "../utils/GtsDict.h"
#include "../utils/RpcUtils.h"
#include "../utils/TypeUtils.h"

namespace milvus {
template class Iterator<SingleResult>;

SearchIteratorImpl::SearchIteratorImpl(MilvusConnectionPtr& connection, const SearchIteratorArguments& args,
                                       const RetryParam& retry_param) {
    connection_ = connection;
    args_ = args;
    retry_param_ = retry_param;

    // might throw exception
    init();
}

Status
SearchIteratorImpl::Next(SingleResult& results) {
    if (reachedLimit()) {
        return Status::OK();
    }

    // how many rows should be outputed in this call?
    auto output_count = args_.BatchSize();
    if (original_limit_ > 0) {
        auto left_count = static_cast<uint64_t>(original_limit_) - returned_count_;
        output_count = std::min(output_count, left_count);
    }

    if (cachedCount() < output_count) {
        // if cache is not sufficient, try to fill the result by probing with constant width
        // until finish filling or exceeding max trial time: 10
        auto status = trySearchFill(output_count);
        if (!status.IsOk()) {
            return status;
        }
    }

    // return batch from the cache if cache is big enough
    auto status = fetchPageFromCache(output_count, results);
    if (!status.IsOk()) {
        return status;
    }

    if (results.GetRowCount() == args_.BatchSize()) {
        updateWidth(results);
    }
    returned_count_ += results.GetRowCount();
    return Status::OK();
}

///////////////////////////////////////////////////////////////////////////////////
// internal methods
void
SearchIteratorImpl::init() {
    original_limit_ = args_.Limit();
    original_params_ = args_.ExtraParams();

    checkOffset();
    checkForSpecialIndexParam();
    checkRangeSearchParameters();
    initSearchIterator();
}

void
SearchIteratorImpl::checkOffset() const {
    if (args_.Offset() > 0) {
        throw std::runtime_error("Not support offset when searching iteration");
    }
}

void
SearchIteratorImpl::checkForSpecialIndexParam() const {
    const auto& params = args_.ExtraParams();
    uint64_t ef = 0;
    auto status = ParseParameter<uint64_t>(params, "ef", ef);
    if (status.IsOk() && ef < args_.BatchSize()) {
        throw std::runtime_error("When using hnsw index, provided ef must be larger than or equal to batch size");
    }
}

void
SearchIteratorImpl::checkRangeSearchParameters() {
    auto metric_type = args_.MetricType();
    if (metric_type == MetricType::DEFAULT) {
        throw std::runtime_error("Must specify metrics type for search iterator");
    }

    const auto& params = args_.ExtraParams();
    double radius = 0.0, range_filter = 0.0;
    auto status_1 = ParseParameter<double>(params, RADIUS, radius);
    auto status_2 = ParseParameter<double>(params, RANGE_FILTER, range_filter);
    if (status_1.IsOk() && status_2.IsOk()) {
        if (metricsPositiveRelated(metric_type) && radius <= range_filter) {
            std::string msg = std::to_string(metric_type) +
                              " metric type, radius must be larger than range_filter, please adjust your parameter";
            throw std::runtime_error(msg);
        }
        if (!metricsPositiveRelated(metric_type) && radius >= range_filter) {
            std::string msg = std::to_string(metric_type) +
                              " metric type, radius must be smalled than range_filter, please adjust your parameter";
            throw std::runtime_error(msg);
        }
    }
}

// L2/JACCARD/HAMMING, smallest value is most similar
// IP/COSINE, largest value is most similar
bool
SearchIteratorImpl::metricsPositiveRelated(MetricType metric_type) const {
    return (metric_type == MetricType::L2 || metric_type == MetricType::JACCARD || metric_type == MetricType::HAMMING);
}

void
SearchIteratorImpl::initSearchIterator() {
    SingleResultPtr single_result;
    auto status = executeSearch(args_.Filter(), false, single_result);
    if (!status.IsOk()) {
        throw std::runtime_error("Fail to init search iterator, error: " + status.Message());
    }
    if (single_result->GetRowCount() == 0) {
        std::string msg = std::string("Cannot init search iterator because init page contains no matched rows, ") +
                          " please check the radius and range_filter set up by searchParams";
        throw std::runtime_error(msg);
    }

    updateWidth(*single_result);
    updateTailDistance(single_result);
    status = updateFilteredIds(single_result);
    if (!status.IsOk()) {
        throw std::runtime_error("Fail to init search iterator, error: " + status.Message());
    }
    cache_.emplace_back(std::move(single_result));
}

// There might be lot of items with the same distance/score value.
// The next search will use the last row's distance/score as the range of search.
// So the next search could return some duplicated ids with the last search.
// This method returns a filter expression to filter out the duplicated ids for the next search.
Status
SearchIteratorImpl::updateFilteredIds(const SingleResultPtr& results) {
    if (results == nullptr || results->GetRowCount() == 0) {
        return Status::OK();
    }

    // scan the results to find out the items whose distance/score is equal to the tail distance
    double last_distance = static_cast<double>(*(results->Scores().rbegin()));
    std::vector<std::string> same_distance_ids;
    const auto& ids = results->Ids();
    const auto& scores = results->Scores();
    for (auto i = 0; i < results->GetRowCount(); i++) {
        if (IsNumEquals(last_distance, static_cast<double>(scores.at(i)))) {
            if (ids.IsIntegerID()) {
                same_distance_ids.push_back(std::to_string(ids.IntIDArray().at(i)));
            } else {
                same_distance_ids.push_back(ids.StrIDArray().at(i));
            }
        }
    }

    if (IsNumEquals(last_distance, tail_distance_)) {
        // if all items have the same distance with the last search, append the ids to filtered_ids_
        // An extreme case:
        //   search_1 returns {1:0.5, 2:0.5, 3:0.5}
        //   search_2 returns {4:0.5, 5:0.5, 6:0.5}
        //   ......
        //   search_N returns {100:0.5, 101:0.5, 102:0.5}
        if (filtered_ids_.empty()) {
            filtered_ids_.swap(same_distance_ids);
        } else {
            std::copy(same_distance_ids.begin(), same_distance_ids.end(), std::back_inserter(filtered_ids_));
        }
    } else {
        // the distance is moved forward, reset the last filtered_ids_
        //   search_1 returns {1:0.5, 2:0.5, 3:0.5}
        //   search_2 returns {4:0.5, 5:0.6, 6:0.6}
        filtered_ids_.swap(same_distance_ids);
    }

    // too many items with the same distance, it is risky to contine iteration
    if (filtered_ids_.size() >= ITERATION_MAX_FILTERED_IDS_COUNT) {
        std::string msg = std::string("filtered ids length has accumulated to more than ") +
                          std::to_string(ITERATION_MAX_FILTERED_IDS_COUNT) +
                          "there is a danger of overly memory consumption";
        return {StatusCode::NOT_SUPPORTED, msg};
    }
    return Status::OK();
}

// this method calculates the upper search range for the next search
void
SearchIteratorImpl::updateTailDistance(const SingleResultPtr& results) {
    if (results == nullptr || results->GetRowCount() == 0) {
        return;
    }

    tail_distance_ = *(results->Scores().rbegin());
}

void
SearchIteratorImpl::updateWidth(const SingleResult& results) {
    if (results.GetRowCount() == 0) {
        return;
    }

    // for L2/JACARRD/HAMMING, smaller distance means more similar, first_distance < last_distance
    // for L2/COSINE, greater distance means more similar, first_distance > last_distance
    // the width_ value is always greater than 0.0
    auto first_distance = *(results.Scores().begin());
    auto last_distance = *(results.Scores().rbegin());
    width_ = std::abs(first_distance - last_distance);

    // enable a minimum value for width to avoid radius and range_filter equal error
    if (width_ <= 0.0) {
        width_ = 0.05;
    }
}

// This method returns a proper limit value for the next search
int64_t
SearchIteratorImpl::extendLimit(bool extend_batch_size) {
    float extend_rate = 1.0;
    if (extend_batch_size) {
        extend_rate = 10.0;
    }
    int64_t next_batch_size = std::min(MAX_BATCH_SIZE, static_cast<int64_t>(args_.BatchSize() * extend_rate));

    // Special process for HNSW index. In HNSW index, limit value cannot be greater than ef value.
    // But if user doesn't input ef value explicitly, the server will use a default ef value.
    // This is a risk that the server might return error since the client don't know the default value,
    // and passes a limit value larger than the default value.
    const auto& params = args_.ExtraParams();
    int64_t ef = 0;
    auto status = ParseParameter<int64_t>(params, "ef", ef);
    if (status.IsOk()) {
        return std::min(next_batch_size, ef);
    }
    return next_batch_size;
}

Status
SearchIteratorImpl::executeSearch(const std::string& filter, bool extend_batch_size, SingleResultPtr& results) {
    uint64_t timeout = connection_->GetConnectParam().RpcDeadlineMs();
    std::string current_db =
        args_.DatabaseName().empty() ? connection_->GetConnectParam().DbName() : args_.DatabaseName();
    proto::milvus::SearchRequest rpc_request;
    auto setParamFunc = [&rpc_request](const std::string& key, const std::string& value) {
        auto kv_pair = rpc_request.add_search_params();
        kv_pair->set_key(key);
        kv_pair->set_value(value);
    };
    setParamFunc(ITERATOR, "True");
    if (args_.CollectionID() > 0) {
        setParamFunc(COLLECTION_ID, std::to_string(args_.CollectionID()));
    }

    // reset the limit value since the iterator fetches data batch by batch
    args_.SetLimit(extendLimit(extend_batch_size));

    auto status = ConvertSearchRequest(args_, current_db, rpc_request);
    if (!status.IsOk()) {
        return status;
    }

    // reset filter, the Next() method will change filter every time
    rpc_request.set_dsl(filter);

    // the Next() will run into this section
    if (session_ts_ > 0) {
        rpc_request.set_guarantee_timestamp(session_ts_);
    }

    // query rpc call via retry process
    proto::milvus::SearchResults rpc_response;
    auto caller = [&]() { return connection_->Search(rpc_request, rpc_response, GrpcOpts{timeout}); };
    status = Retry(caller, retry_param_);
    if (!status.IsOk()) {
        return status;
    }

    if (session_ts_ == 0) {
        // for old milvus versions < 2.5.0, the SearchResults has no session_ts
        // use client-side ts instead
        auto ms = GetNowMs();
        ms = ms << 18;
        session_ts_ = static_cast<uint64_t>(ms);
    }

    SearchResults search_results;
    status = ConvertSearchResults(rpc_response, args_.PkSchema().Name(), search_results);
    if (!status.IsOk()) {
        return status;
    }

    // nq = 1, the search_results must contains a SingleResult. Otherwise it is a server-side bug.
    if (search_results.Results().empty()) {
        return {StatusCode::UNKNOWN_ERROR, "the server returns an empty search result"};
    }

    auto& single_result = search_results.Results().at(0);
    results = std::make_shared<SingleResult>(single_result);
    return Status::OK();
}

bool
SearchIteratorImpl::reachedLimit() const {
    if (original_limit_ > 0 && returned_count_ >= static_cast<uint64_t>(original_limit_)) {
        return true;
    }
    return false;
}

uint64_t
SearchIteratorImpl::cachedCount() const {
    uint64_t cached_count = 0;
    for (const auto& item : cache_) {
        cached_count += item->GetRowCount();
    }
    return cached_count;
}

// Try return a page with count from cache.
// The cache might have multiple SingleResult in a list. This method copy data from
// the header of the list until the count is meet.
Status
SearchIteratorImpl::fetchPageFromCache(int64_t count, SingleResult& results) {
    auto pkField = args_.PkSchema();
    int64_t row_count = 0;

    while (!cache_.empty() && row_count < count) {
        auto one_cache = cache_.front();
        cache_.pop_front();

        if (row_count + one_cache->GetRowCount() <= count) {
            // this batch is smaller than required, append the entire batch
            auto status = AppendSearchResult(*one_cache, results);
            if (!status.IsOk()) {
                return status;
            }
            row_count += one_cache->GetRowCount();
        } else {
            // this batch is greater than required, append a part of the batch
            auto left_count = row_count + one_cache->GetRowCount() - count;
            auto append_count = one_cache->GetRowCount() - left_count;
            std::vector<FieldDataPtr> append_data;
            auto status = CopyFieldsData(one_cache->OutputFields(), 0, append_count, append_data);
            if (!status.IsOk()) {
                return status;
            }

            SingleResult append_result{one_cache->PrimaryKeyName(), one_cache->ScoreName(), std::move(append_data),
                                       args_.OutputFields()};
            status = AppendSearchResult(append_result, results);
            if (!status.IsOk()) {
                return status;
            }
            row_count += append_count;

            // keep the left batch in cache, note we push the left batch in the header of the cache
            if (left_count > 0) {
                std::vector<FieldDataPtr> left_data;
                status = CopyFieldsData(one_cache->OutputFields(), append_count, one_cache->GetRowCount(), left_data);
                if (!status.IsOk()) {
                    return status;
                }
                SingleResultPtr left_batch = std::make_shared<SingleResult>(
                    one_cache->PrimaryKeyName(), one_cache->ScoreName(), std::move(left_data), args_.OutputFields());
                cache_.emplace_front(std::move(left_batch));
            }
        }
    }
    return Status::OK();
}

// This method setups the search range for the next search.
// If user already inputs radius/range, it ensure that the next range is not out of user range.
void
SearchIteratorImpl::nextParams(double range_coefficient) {
    auto metric_type = args_.MetricType();
    double coefficient = std::max(range_coefficient, 1.0);
    double radius = 0.0;
    auto status = ParseParameter<double>(original_params_, RADIUS, radius);
    if (metricsPositiveRelated(metric_type)) {
        double next_radius = tail_distance_ + width_ * coefficient;
        if (status.IsOk() && next_radius > radius) {
            args_.SetRadius(radius);
        } else {
            args_.SetRadius(next_radius);
        }
    } else {
        double next_radius = tail_distance_ - width_ * coefficient;
        if (status.IsOk() && next_radius < radius) {
            args_.SetRadius(radius);
        } else {
            args_.SetRadius(next_radius);
        }
    }
    args_.SetRangeFilter(tail_distance_);
}

std::string
SearchIteratorImpl::filteredDuplicatedResultFilter() {
    if (filtered_ids_.empty()) {
        return args_.Filter();
    }

    std::ostringstream filtered_ids_str;
    bool first = true;
    for (const auto& id : filtered_ids_) {
        if (!first) {
            filtered_ids_str << ", ";
        }
        first = false;
        if (args_.PkSchema().FieldDataType() == DataType::INT64) {
            filtered_ids_str << id;
        } else {
            filtered_ids_str << "'" << id << "'";
        }
    }

    // three cases:
    //  user input expression, no id is filtered out: "name != 'xxx'"
    //  user doesn't input expression, ids are filtered out: "pk not in [3, 4, 5]"
    //  user input expression, ids are filtered out: "name != 'xxx' and pk not in [3, 4, 5]"
    auto and_filter = filtered_ids_str.str();
    if (and_filter.empty()) {
        return args_.Filter();
    } else if (args_.Filter().empty()) {
        return args_.PkSchema().Name() + " not in [" + and_filter + "]";
    } else {
        return args_.Filter() + " and " + args_.PkSchema().Name() + " not in [" + and_filter + "]";
    }
}

// This method will call search multiple times to fill the cache to ensure the cached rows is larger/equal
// than the required count. Each time search is called, the radius/range is enlarged and some ids with the same
// distance will be filtered out.
Status
SearchIteratorImpl::trySearchFill(int64_t count) {
    int try_time = 0;
    double coefficient = 1.0;
    while (true) {
        // setup next search range
        nextParams(coefficient);

        // filter out ids with the same distance with the last row of the last search
        auto next_filter = filteredDuplicatedResultFilter();

        // do search
        SingleResultPtr single_result;
        auto status = executeSearch(next_filter, true, single_result);
        if (!status.IsOk()) {
            return status;
        }
        try_time++;
        // if there's a range containing no vectors matched, then we need to extend
        // the range continually to avoid empty result problem
        coefficient += 1.0;

        if (single_result != nullptr && single_result->GetRowCount() > 0) {
            // ids with the same distance with the last row will be filtered out in the next search
            status = updateFilteredIds(single_result);
            if (!status.IsOk()) {
                return status;
            }

            // setup the next distance for range setting
            updateTailDistance(single_result);

            // cache the rows at the tail of the list
            // warning: single_result will become empty here
            cache_.emplace_back(std::move(single_result));
        }

        // already enough rows
        if (cachedCount() >= count) {
            break;
        }

        if (try_time > ITERATION_MAX_RETRY_TIME) {
            // TODO: a warning log here
            // "Search probe exceed max try times: 20, directly break";
            break;
        }
    }

    return Status::OK();
}

}  // namespace milvus
