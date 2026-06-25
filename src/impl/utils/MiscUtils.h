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

#include "milvus/Status.h"

namespace milvus {

std::string
Trim(const std::string& value);

std::string
UpperWithoutSpaces(const std::string& value);

Status
ParseTargetSizeMB(const std::string& target_size, int64_t& target_size_mb, std::string& normalized);

// Protocol and other machine-readable float text should not depend on the process locale.
bool
ParseFloatWithLocale(const std::string& text, float& value, const std::locale& locale);

}  // namespace milvus
