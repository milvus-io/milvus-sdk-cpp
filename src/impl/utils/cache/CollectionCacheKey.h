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

#include <exception>
#include <functional>
#include <string>

#include "../Uri.h"

namespace milvus {

struct CollectionCacheKey {
    std::string endpoint_;
    std::string db_name_;
    std::string collection_name_;

    bool
    operator==(const CollectionCacheKey& other) const {
        return endpoint_ == other.endpoint_ && db_name_ == other.db_name_ && collection_name_ == other.collection_name_;
    }

    static CollectionCacheKey
    Create(const std::string& endpoint, const std::string& db_name, const std::string& collection_name) {
        return {NormalizeEndpoint(endpoint), db_name.empty() ? "default" : db_name, collection_name};
    }

 private:
    static std::string
    NormalizeEndpoint(const std::string& endpoint) {
        if (endpoint.empty()) {
            return endpoint;
        }

        try {
            const auto uri = ParseURI(endpoint);
            if (uri.host.empty()) {
                return endpoint;
            }

            auto host = uri.host;
            if (host.find(':') != std::string::npos && host.front() != '[') {
                host = "[" + host + "]";
            }
            return host + ":" + std::to_string(uri.port);
        } catch (const std::exception&) {
            return endpoint;
        }
    }
};

struct CollectionCacheKeyHash {
    size_t
    operator()(const CollectionCacheKey& key) const {
        size_t seed = std::hash<std::string>{}(key.endpoint_);
        seed ^= std::hash<std::string>{}(key.db_name_) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<std::string>{}(key.collection_name_) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

}  // namespace milvus
