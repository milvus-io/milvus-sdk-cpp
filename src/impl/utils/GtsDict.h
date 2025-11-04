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

#include <map>
#include <mutex>
#include <string>

namespace milvus {
// GTsDict stores the last write timestamp for ConsistencyLevel.Session
// It is a dict of <string, long>, key is the name of a collection, value is the last write timestamp of the collection.
// It only takes effect when consistency level is Session.
// For each dml action, the GTsDict is updated, the last write timestamp is returned from server-side.
// When search/query/hybridSearch is called, and the consistency level is Session, the ts of the collection will
// be passed to construct a guarantee_ts to the server.
class GtsDict {
 private:
    GtsDict() {
    }

    ~GtsDict() {
    }

 public:
    GtsDict(const GtsDict&) = delete;
    GtsDict&
    operator=(const GtsDict&) = delete;

    static GtsDict&
    GetInstance() {
        static GtsDict instance;
        return instance;
    }

    /**
     * @brief If the collection name exists, use its value to compare to the input ts,
     *  only when the input ts is larger than the existing value, replace it with the input ts.
     *  If the collection name doesn't exist, directly set the input value.
     */
    void
    UpdateCollectionTs(const std::string& db_name, const std::string& collection_name, uint64_t ts);

    /**
     * @brief Get the last write timestamp of a collection.
     *  Return false if the collection name doesn't exist.
     */
    bool
    GetCollectionTs(const std::string& db_name, const std::string& collection_name, uint64_t& ts);

    /**
     * @brief Remove the last write timestamp of a collection.
     */
    void
    RemoveCollectionTs(const std::string& db_name, const std::string& collection_name);

    /**
     * @brief Remove all the timestamps.
     */
    void
    CleanAllCollectionTs();

    static std::string
    CombineName(const std::string& db_name, const std::string& collection_name);

 private:
    std::map<std::string, uint64_t> gts_dict_;
    std::mutex gts_dict_mtx_;
};

int64_t
GetNowMs();

int64_t
MakeMktsFromNowMs();

}  // namespace milvus
