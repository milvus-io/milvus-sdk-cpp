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

#include "milvus/Status.h"
#include "milvus/types/RetryParam.h"

namespace milvus {
Status
Retry(std::function<Status(void)> caller, const RetryParam& retry_param);

// Converts an escaping exception into a Status per the SDK no-throw contract.
// Note: an UNKNOWN_ERROR produced here means an SDK-side exception occurred; if the
// exception escaped from a response-conversion step after the RPC already succeeded,
// the server-side operation may still have been applied. Callers must not treat
// UNKNOWN_ERROR as proof that the operation did not take effect.
Status
StatusFromException(const std::exception& e, const std::string& prefix = "Unexpected SDK exception: ");

// Converts an unknown (non-std::exception) escaping exception into a Status.
Status
StatusFromUnknownException(const std::string& message = "Unexpected SDK exception");

}  // namespace milvus
