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
#include <vector>

namespace milvus {

class ListAliasesResult {
 public:
    ListAliasesResult();
    ListAliasesResult(const std::string& db_name, const std::string& collection_name,
                      const std::vector<std::string>& aliases);

    const std::string&
    GetDbName() const;
    const std::string&
    GetCollectionName() const;
    const std::vector<std::string>&
    GetAliases() const;

    void
    SetDbName(const std::string& db_name);
    void
    SetCollectionName(const std::string& collection_name);
    void
    SetAliases(const std::vector<std::string>& aliases);

 private:
    std::string db_name_{"default"};
    std::string collection_name_;
    std::vector<std::string> aliases_;
};

}  // namespace milvus
