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

#include "milvus/types/SubSearchRequest.h"

#include <memory>

namespace milvus {

SubSearchRequest&
SubSearchRequest::WithMetricType(::milvus::MetricType metric_type) {
    SetMetricType(metric_type);
    return *this;
}

SubSearchRequest&
SubSearchRequest::WithLimit(int64_t limit) {
    SetLimit(limit);
    return *this;
}

SubSearchRequest&
SubSearchRequest::WithFilter(std::string filter) {
    SetFilter(std::move(filter));
    return *this;
}

SubSearchRequest&
SubSearchRequest::WithAnnsField(const std::string& ann_field) {
    SetAnnsField(ann_field);
    return *this;
}

SubSearchRequest&
SubSearchRequest::WithTimezone(const std::string& timezone) {
    SetTimezone(timezone);
    return *this;
}

}  // namespace milvus
