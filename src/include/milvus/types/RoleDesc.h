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

struct Privilege {
    std::string object_type;
    std::string object_name;
    std::string db_name;
    std::string role_name;
    std::string privilege;
    std::string grantor_name;
};

class RoleDesc {
 public:
    RoleDesc();
    RoleDesc(const std::string& role, const std::vector<Privilege>& privileges);

    const std::string&
    GetRole() const;
    const std::vector<Privilege>&
    GetPrivileges() const;

 private:
    std::string role_;
    std::vector<Privilege> privileges_;
};

}  // namespace milvus
