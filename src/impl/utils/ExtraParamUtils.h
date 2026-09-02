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
#include <unordered_map>

#include "./TypeUtils.h"

namespace milvus {

using ExtraParamsMap = std::unordered_map<std::string, std::string>;

inline void
SetExtraInt64(ExtraParamsMap& params, const std::string& key, int64_t val) {
    params[key] = std::to_string(val);
}

inline int64_t
GetExtraInt64(const ExtraParamsMap& params, const std::string& key, int64_t default_val) {
    auto it = params.find(key);
    if (it != params.end()) {
        return std::stoll(it->second);
    }
    return default_val;
}

inline bool
TryGetExtraInt64(const ExtraParamsMap& params, const std::string& key, int64_t& value) {
    auto it = params.find(key);
    if (it == params.end()) {
        return false;
    }
    try {
        value = std::stoll(it->second);
        return true;
    } catch (...) {
        return false;
    }
}

inline Status
ValidateLimit(int64_t limit) {
    if (limit <= 0) {
        return {StatusCode::INVALID_ARGUMENT, "limit must be greater than 0"};
    }
    return Status::OK();
}

inline Status
ValidateRoundDecimal(const ExtraParamsMap& params) {
    int64_t round_decimal = -1;
    if (TryGetExtraInt64(params, "round_decimal", round_decimal)) {
        if (round_decimal <= -2 || round_decimal >= 7) {
            return {StatusCode::INVALID_ARGUMENT, "round_decimal must be in range (-2, 7)"};
        }
        return Status::OK();
    }
    if (params.count("round_decimal") != 0) {
        return {StatusCode::INVALID_ARGUMENT, "round_decimal must be a valid integer"};
    }
    return Status::OK();
}

inline std::string
DoubleToString(double val) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(15) << val;
    return stream.str();
}

inline void
SetExtraDouble(ExtraParamsMap& params, const std::string& key, double val) {
    params[key] = DoubleToString(val);
}

inline double
GetExtraDouble(const ExtraParamsMap& params, const std::string& key, double default_val) {
    auto it = params.find(key);
    if (it != params.end()) {
        return std::stod(it->second);
    }
    return default_val;
}

inline void
SetExtraBool(ExtraParamsMap& params, const std::string& key, bool val) {
    params[key] = val ? "true" : "false";
}

inline bool
GetExtraBool(const ExtraParamsMap& params, const std::string& key, bool default_val) {
    auto it = params.find(key);
    if (it != params.end()) {
        return it->second == "true" ? true : false;
    }
    return default_val;
}

inline void
SetExtraStr(ExtraParamsMap& params, const std::string& key, const std::string& val) {
    params[key] = val;
}

inline std::string
GetExtraStr(const ExtraParamsMap& params, const std::string& key, std::string default_val) {
    auto it = params.find(key);
    if (it != params.end()) {
        return it->second;
    }
    return default_val;
}

template <typename T>
Status
ParseParameter(const std::unordered_map<std::string, std::string>& params, const std::string& name, T& value) {
    auto iter = params.find(name);
    if (iter == params.end()) {
        return {StatusCode::INVALID_ARGUMENT, "no such parameter"};
    }
    try {
        if (std::is_integral<T>::value) {
            value = static_cast<T>(std::stol(iter->second));
        } else if (std::is_floating_point<T>::value) {
            value = static_cast<T>(std::stod(iter->second));
        } else {
            return {StatusCode::INVALID_ARGUMENT, "can only parse integer and float type value"};
        }
    } catch (...) {
        return {StatusCode::INVALID_ARGUMENT, "parameter '" + name + "' value '" + iter->second + "' cannot be parsed"};
    }
    return Status::OK();
}

}  // namespace milvus
