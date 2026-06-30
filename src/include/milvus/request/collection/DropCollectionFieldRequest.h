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

#include "./CollectionRequestBase.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::DropCollectionField()
 */
class MILVUS_SDK_API DropCollectionFieldRequest : public CollectionRequestBase<DropCollectionFieldRequest> {
 public:
    /**
     * @brief Constructor
     */
    DropCollectionFieldRequest() = default;

    /**
     * @brief Name of the field to drop.
     */
    const std::string&
    FieldName() const;

    /**
     * @brief Set the name of the field to drop.
     */
    void
    SetFieldName(std::string field_name);

    /**
     * @brief Set the name of the field to drop.
     */
    DropCollectionFieldRequest&
    WithFieldName(std::string field_name);

    /**
     * @brief ID of the field to drop.
     */
    int64_t
    FieldID() const;

    /**
     * @brief Set the ID of the field to drop.
     */
    void
    SetFieldID(int64_t field_id);

    /**
     * @brief Set the ID of the field to drop.
     */
    DropCollectionFieldRequest&
    WithFieldID(int64_t field_id);

 private:
    std::string field_name_;
    int64_t field_id_{0};
};

}  // namespace milvus
