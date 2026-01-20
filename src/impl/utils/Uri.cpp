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

#include "./Uri.h"

namespace milvus {

URI
ParseURI(const std::string& url) {
    URI out;

    // scheme
    const std::string scheme_sep = "://";
    const auto scheme_pos = url.find(scheme_sep);
    if (scheme_pos != std::string::npos) {
        out.scheme = url.substr(0, scheme_pos);
    }

    // authority start
    const std::size_t auth_start = (scheme_pos == std::string::npos) ? 0 : scheme_pos + scheme_sep.size();

    // path start (first '/' after authority)
    const auto path_pos = url.find('/', auth_start);
    const std::size_t auth_end = (path_pos == std::string::npos) ? url.size() : path_pos;

    // authority = host[:port]  (basic)
    const std::string authority = url.substr(auth_start, auth_end - auth_start);

    // host + port
    if (!authority.empty()) {
        // IPv6 in brackets: [::1]:19530
        if (authority.front() == '[') {
            const auto close = authority.find(']');
            if (close != std::string::npos) {
                out.host = authority.substr(1, close - 1);
                if (close + 1 < authority.size() && authority[close + 1] == ':') {
                    const std::string port_str = authority.substr(close + 2);
                    out.port = port_str.empty() ? -1 : static_cast<uint16_t>(std::stoi(port_str));
                }
            } else {
                // malformed; treat whole authority as host
                out.host = authority;
            }
        } else {
            // Normal host:port
            const auto colon = authority.rfind(':');
            if (colon != std::string::npos && authority.find(':') == colon) {
                // exactly one ':' => interpret as host:port
                out.host = authority.substr(0, colon);
                const std::string port_str = authority.substr(colon + 1);
                out.port = port_str.empty() ? -1 : static_cast<uint16_t>(std::stoi(port_str));
            } else {
                // no port (or multiple ':' likely IPv6 without brackets)
                out.host = authority;
                out.port = (out.scheme == "https" ? 443 : 19530);
            }
        }
    }

    // path
    out.path = (path_pos == std::string::npos) ? "" : url.substr(path_pos);

    // Extract db_name from URI path only if appropriate
    // 1. If db_name is empty string and URI has path -> use URI path
    // 2. If db_name is empty string and URI has no path -> use default db
    if (out.path.empty() || out.path == "/") {
        // keep default dbname
    } else {
        // substring(1): drop leading '/'
        out.dbname = out.path.substr(1);
    }

    return out;
}

}  // namespace milvus
