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

#include "milvus/types/IteratorArguments.h"

#include "../utils/Constants.h"
#include "milvus/types/Iterator.h"

namespace milvus {

/////////////////////////////////////////////////////////////////////////////////////
// QueryIteratorCursor
QueryIteratorCursor::QueryIteratorCursor() = default;

uint64_t
QueryIteratorCursor::SessionTs() const {
    return session_ts_;
}

Status
QueryIteratorCursor::SetSessionTs(uint64_t session_ts) {
    session_ts_ = session_ts;
    return Status::OK();
}

QueryIteratorCursor&
QueryIteratorCursor::WithSessionTs(uint64_t session_ts) {
    session_ts_ = session_ts;
    return *this;
}

bool
QueryIteratorCursor::HasIntPk() const {
    return has_int_pk_;
}

int64_t
QueryIteratorCursor::IntPk() const {
    return int_pk_;
}

Status
QueryIteratorCursor::SetIntPk(int64_t pk) {
    int_pk_ = pk;
    has_int_pk_ = true;
    has_str_pk_ = false;
    str_pk_.clear();
    return Status::OK();
}

QueryIteratorCursor&
QueryIteratorCursor::WithIntPk(int64_t pk) {
    SetIntPk(pk);
    return *this;
}

bool
QueryIteratorCursor::HasStrPk() const {
    return has_str_pk_;
}

const std::string&
QueryIteratorCursor::StrPk() const {
    return str_pk_;
}

Status
QueryIteratorCursor::SetStrPk(const std::string& pk) {
    str_pk_ = pk;
    has_str_pk_ = true;
    has_int_pk_ = false;
    int_pk_ = 0;
    return Status::OK();
}

QueryIteratorCursor&
QueryIteratorCursor::WithStrPk(const std::string& pk) {
    SetStrPk(pk);
    return *this;
}

bool
QueryIteratorCursor::HasLastElementOffset() const {
    return has_last_element_offset_;
}

int64_t
QueryIteratorCursor::LastElementOffset() const {
    return last_element_offset_;
}

Status
QueryIteratorCursor::SetLastElementOffset(int64_t offset) {
    if (offset < 0) {
        return {StatusCode::INVALID_ARGUMENT, "last element offset must not be negative"};
    }
    last_element_offset_ = offset;
    has_last_element_offset_ = true;
    return Status::OK();
}

QueryIteratorCursor&
QueryIteratorCursor::WithLastElementOffset(int64_t offset) {
    SetLastElementOffset(offset);
    return *this;
}

int64_t
IteratorArguments::BatchSize() const {
    return batch_size_;
}

Status
IteratorArguments::SetBatchSize(int64_t batch_size) {
    if (batch_size == 0) {
        return {StatusCode::INVALID_ARGUMENT, "batch size must be greater than zero"};
    }
    if (batch_size > MAX_BATCH_SIZE) {
        return {StatusCode::INVALID_ARGUMENT, "batch size cannot be larger than " + std::to_string(MAX_BATCH_SIZE)};
    }

    batch_size_ = batch_size;
    return Status::OK();
}

int64_t
IteratorArguments::CollectionID() const {
    return collection_id_;
}

Status
IteratorArguments::SetCollectionID(int64_t id) {
    collection_id_ = id;
    return Status::OK();
}

const FieldSchema&
IteratorArguments::PkSchema() const {
    return pk_schema_;
}

Status
IteratorArguments::SetPkSchema(const FieldSchema& schema) {
    pk_schema_ = schema;
    return Status::OK();
}

const std::function<Status(SingleResult&)>&
IteratorArguments::ExternalFilterFunc() const {
    return external_filter_func_;
}

void
IteratorArguments::SetExternalFilterFunc(const std::function<Status(SingleResult&)>& func) {
    external_filter_func_ = func;
}

/////////////////////////////////////////////////////////////////////////////////////
// QueryIteratorArguments
bool
QueryIteratorArguments::ReduceStopForBest() const {
    return reduce_stop_for_best_;
}

Status
QueryIteratorArguments::SetReduceStopForBest(bool reduce_stop_for_best) {
    reduce_stop_for_best_ = reduce_stop_for_best;
    return Status::OK();
}

}  // namespace milvus
