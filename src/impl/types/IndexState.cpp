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

#include "milvus/types/IndexState.h"

namespace milvus {

IndexStateCode
IndexState::StateCode() const {
    return state_code_;
}

void
IndexState::SetStateCode(IndexStateCode state_code) {
    state_code_ = state_code;
}

std::string
IndexState::FailedReason() const {
    return failed_reason_;
}
void
IndexState::SetFailedReason(std::string failed_reason) {
    failed_reason_ = std::move(failed_reason);
}

int64_t
IndexProgress::IndexedRows() const {
    return indexed_rows_;
}

void
IndexProgress::SetIndexedRows(int64_t indexed_rows) {
    indexed_rows_ = indexed_rows;
}

int64_t
IndexProgress::TotalRows() const {
    return total_rows_;
}

void
IndexProgress::SetTotalRows(int64_t total_rows) {
    total_rows_ = total_rows;
}

}  // namespace milvus
