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

#pragma once
#include <cstdint>
#include <string>

namespace milvus {

inline const std::string&
KeyTopK() {
    static std::string topk = "topk";
    return topk;
}

inline const std::string&
KeyLimit() {
    static std::string limit = "limit";
    return limit;
}

inline const std::string&
KeyOffset() {
    static std::string offset = "offset";
    return offset;
}

inline const std::string&
KeyAnnsField() {
    static std::string anns_field = "anns_field";
    return anns_field;
}

inline const std::string&
KeyRadius() {
    static std::string radius = "radius";
    return radius;
}

inline const std::string&
KeyRangeFilter() {
    static std::string range_filter = "range_filter";
    return range_filter;
}

inline const std::string&
KeyIgnoreGrowing() {
    static std::string ignore_growing = "ignore_growing";
    return ignore_growing;
}

inline const std::string&
KeyParams() {
    static std::string params = "params";
    return params;
}

inline const std::string&
KeyStrategy() {
    static std::string strategy = "strategy";
    return strategy;
}

inline uint64_t
GuaranteeStrongTs() {
    return 0;
}

inline uint64_t
GuaranteeEventuallyTs() {
    return 1;
}

}  // namespace milvus
