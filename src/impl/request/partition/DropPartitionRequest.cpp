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

#include "milvus/request/partition/DropPartitionRequest.h"

namespace milvus {

DropPartitionRequest&
DropPartitionRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

DropPartitionRequest&
DropPartitionRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

DropPartitionRequest&
DropPartitionRequest::WithPartitionName(const std::string& partition_name) {
    SetPartitionName(partition_name);
    return *this;
}

}  // namespace milvus
