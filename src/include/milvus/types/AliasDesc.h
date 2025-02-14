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

#include <string>

namespace milvus {

class AliasDesc {
 public:
    AliasDesc();
    AliasDesc(const std::string& db_name, const std::string& alias, const std::string& collection_name);

    const std::string&
    GetDbName() const;
    const std::string&
    GetAlias() const;
    const std::string&
    GetCollectionName() const;

    void
    SetDbName(const std::string& db_name);
    void
    SetAlias(const std::string& alias);
    void
    SetCollectionName(const std::string& collection_name);

 private:
    std::string db_name_{"default"};
    std::string alias_;
    std::string collection_name_;
};

}  // namespace milvus
