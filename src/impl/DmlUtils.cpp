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

#include "DmlUtils.h"

#include "GtsDict.h"
#include "milvus/types/Constants.h"

namespace milvus {

bool
IsInputField(const FieldSchema& field_schema, bool is_upsert) {
    // in v2.4, all the fields except the auto-id field are required for insert()
    // but in upsert(), all the fields including the auto-id field are requred to input
    if (field_schema.IsPrimaryKey() && field_schema.AutoID()) {
        return is_upsert;
    }
    // dynamic field is optional, not required by force
    if (field_schema.Name() == DynamicFieldName()) {
        return false;
    }
    return true;
}

// The returned status error code affects the collection schema cache in MilvusClientImpl,
// carefully return the error code for different cases.
// DATA_UNMATCH_SCHEMA will tell the MilvusClientImpl to update collection schema cache,
// and call CheckInsertInput() to check the input again.
// Other error codes will be treated as failure immediatelly.
Status
CheckInsertInput(const CollectionDescPtr& collection_desc, const std::vector<FieldDataPtr>& fields, bool is_upsert) {
    bool enable_dynamic_field = collection_desc->Schema().EnableDynamicField();
    const auto& collection_fields = collection_desc->Schema().Fields();

    // this loop is for "are there any redundant data?"
    for (const auto& field : fields) {
        auto it = std::find_if(collection_fields.begin(), collection_fields.end(),
                               [&field](const FieldSchema& schema) { return schema.Name() == field->Name(); });
        if (it != collection_fields.end()) {
            // the provided field is in collection schema, but it is not a required input
            // maybe the schema has been changed(primary key from auto-id to non-auto-id)
            // tell the MilvusClientImpl to update collection schema cache
            if (!IsInputField(*it, is_upsert)) {
                return Status{StatusCode::DATA_UNMATCH_SCHEMA,
                              std::string(field->Name() + " is auto-id field, no need to provide")};
            }
            // accept it
            continue;
        }
        if (field->Name() == DynamicFieldName()) {
            // if dynamic field is not JSON type, no need to update collection schema cache
            if (field->Type() != DataType::JSON) {
                return Status{StatusCode::INVALID_AGUMENT,
                              std::string(field->Name() + " is name of dynamic field, the field type must be JSON")};
            }
            // if has dynamic field data but enable_dynamic_field is false, maybe the schema cache is out of date
            if (!enable_dynamic_field) {
                return Status{StatusCode::DATA_UNMATCH_SCHEMA, std::string(field->Name() + " is not a valid field")};
            }
            // enable_dynamic_field is true and has dynamic field data
            // maybe the schema cache is out of date(enable_dynamic_field from true to false)
            // but we don't know, just pass the data to the server to check
            continue;
        }

        // redundant fields, maybe the schema has been changed(some fields added)
        // tell the MilvusClientImpl to update collection schema cache
        return Status{StatusCode::DATA_UNMATCH_SCHEMA, std::string(field->Name() + " is not a valid field")};
    }

    // this loop is for "are there any data missed?
    for (const auto& collection_field : collection_fields) {
        auto it = std::find_if(fields.begin(), fields.end(), [&collection_field](const FieldDataPtr& field) {
            return field->Name() == collection_field.Name();
        });

        if (it != fields.end()) {
            continue;
        }

        // some required fields are not provided, maybe the schema has been changed(some fields deleted)
        // tell the MilvusClientImpl to update collection schema cache
        if (IsInputField(collection_field, is_upsert)) {
            return Status{StatusCode::DATA_UNMATCH_SCHEMA,
                          std::string("data of the field " + collection_field.Name() + " is missed")};
        }
    }
    return Status::OK();
}

bool
IsRealFailure(const proto::common::Status& status) {
    // error_code() is legacy code, deprecated in v2.4, code() is new code returned by higher version milvus
    // both error_code() == RateLimit or code() == 8 means rate limit error
    return ((status.error_code() != proto::common::ErrorCode::RateLimit) &&
            (status.error_code() != proto::common::ErrorCode::Success)) ||
           (status.code() != 0 && status.code() != 8);
}

uint64_t
DeduceGuaranteeTimestamp(const ConsistencyLevel& level, const std::string& db_name,
                         const std::string& collection_name) {
    if (level == ConsistencyLevel::NONE) {
        uint64_t ts = 1;
        return GtsDict::GetInstance().GetCollectionTs(db_name, collection_name, ts) ? ts : 1;
    }

    switch (level) {
        case ConsistencyLevel::STRONG:
            return 0;
        case ConsistencyLevel::SESSION: {
            uint64_t ts = 1;
            return GtsDict::GetInstance().GetCollectionTs(db_name, collection_name, ts) ? ts : 1;
        }
        case ConsistencyLevel::BOUNDED:
            return 2;  // let server side to determine the bounded time
        default:
            return 1;  // EVENTUALLY and others
    }
}

}  // namespace milvus
