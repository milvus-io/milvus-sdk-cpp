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

#include "MiscUtils.h"

#include <cctype>
#include <cmath>
#include <limits>
#include <locale>
#include <sstream>

namespace {

bool
ParsePositiveDecimal(const std::string& text, long double& number, size_t& unit_pos) {
    size_t pos = 0;
    bool negative = false;
    if (pos < text.size() && (text[pos] == '+' || text[pos] == '-')) {
        negative = text[pos] == '-';
        ++pos;
    }

    bool has_digit = false;
    long double value = 0;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
        has_digit = true;
        value = value * 10 + (text[pos] - '0');
        ++pos;
    }

    if (pos < text.size() && text[pos] == '.') {
        ++pos;
        long double scale = 0.1L;
        while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
            has_digit = true;
            value += (text[pos] - '0') * scale;
            scale /= 10;
            ++pos;
        }
    }

    if (!has_digit || negative) {
        return false;
    }

    if (pos < text.size() && (text[pos] == 'e' || text[pos] == 'E')) {
        auto exponent_pos = pos++;
        bool exponent_negative = false;
        if (pos < text.size() && (text[pos] == '+' || text[pos] == '-')) {
            exponent_negative = text[pos] == '-';
            ++pos;
        }

        bool has_exponent_digit = false;
        int exponent = 0;
        while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
            has_exponent_digit = true;
            if (exponent < 10000) {
                exponent = exponent * 10 + (text[pos] - '0');
            }
            ++pos;
        }
        if (!has_exponent_digit) {
            pos = exponent_pos;
        } else {
            value *= std::pow(10.0L, exponent_negative ? -exponent : exponent);
        }
    }

    number = value;
    unit_pos = pos;
    return std::isfinite(number) && number > 0;
}

}  // namespace

namespace milvus {

std::string
Trim(const std::string& value) {
    auto first = value.begin();
    while (first != value.end() && std::isspace(static_cast<unsigned char>(*first))) {
        ++first;
    }

    auto last = value.end();
    while (last != first && std::isspace(static_cast<unsigned char>(*(last - 1)))) {
        --last;
    }

    return std::string(first, last);
}

std::string
UpperWithoutSpaces(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (auto ch : value) {
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            result.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
        }
    }
    return result;
}

bool
ParseFloatWithLocale(const std::string& text, float& value, const std::locale& locale) {
    std::istringstream stream(text);
    stream.imbue(locale);
    stream >> std::noskipws >> value;
    return stream && stream.eof();
}

Status
ParseTargetSizeMB(const std::string& target_size, int64_t& target_size_mb, std::string& normalized) {
    target_size_mb = 0;
    normalized.clear();

    auto text = Trim(target_size);
    if (text.empty()) {
        return Status::OK();
    }

    long double number = 0;
    size_t unit_pos = 0;
    if (!ParsePositiveDecimal(text, number, unit_pos)) {
        return {StatusCode::INVALID_ARGUMENT, "Invalid optimize target size: " + target_size};
    }

    auto unit = UpperWithoutSpaces(text.substr(unit_pos));
    long double multiplier = 1.0L;
    if (unit.empty() || unit == "B") {
        multiplier = 1.0L;
    } else if (unit == "KB") {
        multiplier = 1024.0L;
    } else if (unit == "MB") {
        multiplier = 1024.0L * 1024.0L;
    } else if (unit == "GB") {
        multiplier = 1024.0L * 1024.0L * 1024.0L;
    } else if (unit == "TB") {
        multiplier = 1024.0L * 1024.0L * 1024.0L * 1024.0L;
    } else if (unit == "PB") {
        multiplier = 1024.0L * 1024.0L * 1024.0L * 1024.0L * 1024.0L;
    } else {
        return {StatusCode::INVALID_ARGUMENT, "Invalid optimize target size unit: " + target_size};
    }

    const auto bytes = number * multiplier;
    constexpr long double mb_bytes = 1024.0L * 1024.0L;
    if (bytes < mb_bytes || bytes > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        return {StatusCode::INVALID_ARGUMENT, "Optimize target size must be at least 1MB"};
    }

    target_size_mb = static_cast<int64_t>(bytes / mb_bytes);
    if (target_size_mb <= 0) {
        return {StatusCode::INVALID_ARGUMENT, "Optimize target size must be at least 1MB"};
    }
    normalized = std::to_string(target_size_mb) + "MB";
    return Status::OK();
}

}  // namespace milvus
