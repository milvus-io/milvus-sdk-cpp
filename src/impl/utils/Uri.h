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

struct URI {
    std::string scheme;  // e.g. "http"
    std::string host;    // hostname only (no brackets for IPv6)
    uint16_t port = 0;   // 0 if not present
    std::string path;    // e.g. "/mydb" or "/"
    std::string dbname;  // path without leading '/', or ""
};

URI
ParseURI(const std::string& url);

}  // namespace milvus
