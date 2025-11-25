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

#include "milvus/request/index/DescribeIndexRequest.h"

namespace milvus {

DescribeIndexRequest&
DescribeIndexRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

DescribeIndexRequest&
DescribeIndexRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

const std::string&
DescribeIndexRequest::FieldName() const {
    return field_name_;
}

void
DescribeIndexRequest::SetFieldName(const std::string& field_name) {
    field_name_ = field_name;
}

DescribeIndexRequest&
DescribeIndexRequest::WithFieldName(const std::string& field_name) {
    SetFieldName(field_name);
    return *this;
}

int64_t
DescribeIndexRequest::Timestamp() const {
    return timestamp_;
}

void
DescribeIndexRequest::SetTimestamp(int64_t ts) {
    timestamp_ = ts;
}

DescribeIndexRequest&
DescribeIndexRequest::WithTimestamp(int64_t ts) {
    SetTimestamp(ts);
    return *this;
}

}  // namespace milvus
