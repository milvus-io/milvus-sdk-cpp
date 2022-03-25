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

#include "milvus/types/HybirdTimestamp.h"

namespace milvus {

HybirdTimestamp::HybirdTimestamp() = default;

HybirdTimestamp::HybirdTimestamp(uint64_t ts) : ts_(ts) {
}

HybirdTimestamp::HybirdTimestamp(uint64_t physical, uint64_t logical)
    : ts_((physical << milvus::HybirdtsLogicalBits()) + logical) {
}

uint64_t
HybirdTimestamp::Timestamp() const {
    return ts_;
}

uint64_t
HybirdTimestamp::Logical() const {
    return ts_ & milvus::HybirdtsLogicalBitsMask();
}

uint64_t
HybirdTimestamp::Physical() const {
    return ts_ >> milvus::HybirdtsLogicalBits();
}

HybirdTimestamp&
HybirdTimestamp::operator+=(uint64_t milliseconds) {
    ts_ += (milliseconds << milvus::HybirdtsLogicalBits());
    return *this;
}

HybirdTimestamp
HybirdTimestamp::operator+(uint64_t milliseconds) {
    return {Physical() + milliseconds, Logical()};
}

HybirdTimestamp
HybirdTimestamp::CreateFromUnixtime(uint64_t epoch_in_milliseconds) {
    return {epoch_in_milliseconds, 0};
}
}  // namespace milvus
