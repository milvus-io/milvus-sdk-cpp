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

#include "milvus/types/IDArray.h"

namespace milvus {

IDArray::IDArray(const std::vector<int64_t>& id_array) : int_id_array_(id_array) {
}

IDArray::IDArray(std::vector<int64_t>&& id_array) : int_id_array_(std::move(id_array)) {
}

IDArray::IDArray(const std::vector<std::string>& id_array) : str_id_array_(id_array), is_int_array_{false} {
}

IDArray::IDArray(std::vector<std::string>&& id_array) : str_id_array_(std::move(id_array)), is_int_array_{false} {
}

bool
IDArray::IsIntegerID() const {
    return is_int_array_;
}

const std::vector<int64_t>&
IDArray::IntIDArray() const {
    return int_id_array_;
}

const std::vector<std::string>&
IDArray::StrIDArray() const {
    return str_id_array_;
}

}  // namespace milvus
