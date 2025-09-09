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

namespace milvus {
uint64_t
IteratorArguments::BatchSize() const {
    return batch_size_;
}

Status
IteratorArguments::SetBatchSize(uint64_t batch_size) {
    if (batch_size == 0) {
        return {StatusCode::INVALID_AGUMENT, "batch size must be greater than zero"};
    }
    if (batch_size > MAX_BATCH_SIZE) {
        return {StatusCode::INVALID_AGUMENT, "batch size cannot be larger than " + std::to_string(MAX_BATCH_SIZE)};
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
