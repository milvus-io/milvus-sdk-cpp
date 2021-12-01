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

#include <memory>

#include "Status.h"
#include "types/CollectionDesc.h"
#include "types/CollectionInfo.h"
#include "types/CollectionSchema.h"
#include "types/CollectionStat.h"
#include "types/ConnectParam.h"
#include "types/TimeoutSetting.h"

/**
 *  @brief namespace milvus
 */
namespace milvus {

/**
 * @brief milvus client abstract class
 */
class MilvusClient {
 public:
    static std::shared_ptr<MilvusClient>
    Create();

    virtual Status
    Connect(const ConnectParam& connect_param) = 0;

    virtual Status
    Disconnect() = 0;

    virtual Status
    CreateCollection(const CollectionSchema& schema) = 0;

    virtual Status
    HasCollection(const std::string& collection_name, bool& has) = 0;

    virtual Status
    DropCollection(const std::string& collection_name) = 0;

    virtual Status
    LoadCollection(const std::string& collection_name, const TimeoutSetting* timeout) = 0;

    virtual Status
    ReleaseCollection(const std::string& collection_name) = 0;

    virtual Status
    DescribeCollection(const std::string& collection_name, CollectionDesc& collection_desc) = 0;

    virtual Status
    GetCollectionStatistics(const std::string& collection_name, bool do_flush, CollectionStat& collection_stat) = 0;

    virtual Status
    ShowCollections(const std::vector<std::string>& collection_names, CollectionsInfo& collection_desc) = 0;
};

}  // namespace milvus
