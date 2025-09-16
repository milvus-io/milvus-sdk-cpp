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

#include "milvus/types/DmlResults.h"

namespace milvus {

const IDArray&
DmlResults::IdArray() const {
    return id_array_;
}

void
DmlResults::SetIdArray(const IDArray& id_array) {
    id_array_ = id_array;
}

void
DmlResults::SetIdArray(IDArray&& id_array) {
    id_array_ = id_array;
}

uint64_t
DmlResults::Timestamp() const {
    return timestamp_;
}

void
DmlResults::SetTimestamp(uint64_t timestamp) {
    timestamp_ = timestamp;
}

uint64_t
DmlResults::InsertCount() const {
    return insert_cnt_;
}

void
DmlResults::SetInsertCount(uint64_t count) {
    insert_cnt_ = count;
}

uint64_t
DmlResults::DeleteCount() const {
    return delete_cnt_;
}

void
DmlResults::SetDeleteCount(uint64_t count) {
    delete_cnt_ = count;
}

uint64_t
DmlResults::UpsertCount() const {
    return upsert_cnt_;
}

void
DmlResults::SetUpsertCount(uint64_t count) {
    upsert_cnt_ = count;
}

}  // namespace milvus
