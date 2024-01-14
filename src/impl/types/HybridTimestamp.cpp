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

#include "milvus/types/HybridTimestamp.h"

namespace milvus {

HybridTimestamp::HybridTimestamp() = default;

HybridTimestamp::HybridTimestamp(uint64_t ts) : ts_(ts) {
}

HybridTimestamp::HybridTimestamp(uint64_t physical, uint64_t logical)
    : ts_((physical << milvus::HybridTsLogicalBits()) + logical) {
}

uint64_t
HybridTimestamp::Timestamp() const {
    return ts_;
}

uint64_t
HybridTimestamp::Logical() const {
    return ts_ & milvus::HybridTsLogicalBitsMask();
}

uint64_t
HybridTimestamp::Physical() const {
    return ts_ >> milvus::HybridTsLogicalBits();
}

HybridTimestamp&
HybridTimestamp::operator+=(uint64_t milliseconds) {
    ts_ += (milliseconds << milvus::HybridTsLogicalBits());
    return *this;
}

HybridTimestamp
HybridTimestamp::operator+(uint64_t milliseconds) const {
    return {Physical() + milliseconds, Logical()};
}

HybridTimestamp
HybridTimestamp::CreateFromUnixTime(uint64_t epoch_in_milliseconds) {
    return {epoch_in_milliseconds, 0};
}
}  // namespace milvus
