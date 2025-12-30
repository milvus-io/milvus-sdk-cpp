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

#include "milvus/request/collection/CreateSimpleCollectionRequest.h"

namespace milvus {

const std::string&
CreateSimpleCollectionRequest::DatabaseName() const {
    return db_name_;
}

void
CreateSimpleCollectionRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

CreateSimpleCollectionRequest&
CreateSimpleCollectionRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

const std::string&
CreateSimpleCollectionRequest::CollectionName() const {
    return collection_name_;
}

void
CreateSimpleCollectionRequest::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

CreateSimpleCollectionRequest&
CreateSimpleCollectionRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

const std::string&
CreateSimpleCollectionRequest::PrimaryFieldName() const {
    return primary_field_name_;
}

void
CreateSimpleCollectionRequest::SetPrimaryFieldName(const std::string& primary_field_name) {
    primary_field_name_ = primary_field_name;
}

CreateSimpleCollectionRequest&
CreateSimpleCollectionRequest::WithPrimaryFieldName(const std::string& primary_field_name) {
    SetPrimaryFieldName(primary_field_name);
    return *this;
}

DataType
CreateSimpleCollectionRequest::PrimaryFieldType() const {
    return primary_field_type_;
}

void
CreateSimpleCollectionRequest::SetPrimaryFieldType(DataType primary_field_type) {
    primary_field_type_ = primary_field_type;
}

CreateSimpleCollectionRequest&
CreateSimpleCollectionRequest::WithPrimaryFieldType(DataType primary_field_type) {
    SetPrimaryFieldType(primary_field_type);
    return *this;
}

const std::string&
CreateSimpleCollectionRequest::VectorFieldName() const {
    return vector_field_name_;
}

void
CreateSimpleCollectionRequest::SetVectorFieldName(const std::string& vector_field_name) {
    vector_field_name_ = vector_field_name;
}

CreateSimpleCollectionRequest&
CreateSimpleCollectionRequest::WithVectorFieldName(const std::string& vector_field_name) {
    SetVectorFieldName(vector_field_name);
    return *this;
}

int64_t
CreateSimpleCollectionRequest::Dimension() const {
    return dimension_;
}

void
CreateSimpleCollectionRequest::SetDimension(int64_t dimension) {
    dimension_ = dimension;
}

CreateSimpleCollectionRequest&
CreateSimpleCollectionRequest::WithDimension(int64_t dimension) {
    SetDimension(dimension);
    return *this;
}

milvus::ConsistencyLevel
CreateSimpleCollectionRequest::ConsistencyLevel() const {
    return level_;
}

void
CreateSimpleCollectionRequest::SetConsistencyLevel(milvus::ConsistencyLevel level) {
    level_ = level;
}

CreateSimpleCollectionRequest&
CreateSimpleCollectionRequest::WithConsistencyLevel(milvus::ConsistencyLevel level) {
    SetConsistencyLevel(level);
    return *this;
}

milvus::MetricType
CreateSimpleCollectionRequest::MetricType() const {
    return metric_type_;
}

void
CreateSimpleCollectionRequest::SetMetricType(milvus::MetricType metric_type) {
    metric_type_ = metric_type;
}

CreateSimpleCollectionRequest&
CreateSimpleCollectionRequest::WithMetricType(milvus::MetricType metric_type) {
    SetMetricType(metric_type);
    return *this;
}

bool
CreateSimpleCollectionRequest::AutoID() const {
    return auto_id_;
}

void
CreateSimpleCollectionRequest::SetAutoID(bool auto_id) {
    auto_id_ = auto_id;
}

CreateSimpleCollectionRequest&
CreateSimpleCollectionRequest::WithAutoID(bool auto_id) {
    SetAutoID(auto_id);
    return *this;
}

bool
CreateSimpleCollectionRequest::EnableDynamicField() const {
    return enable_dynamic_field_;
}

void
CreateSimpleCollectionRequest::SetEnableDynamicField(bool enable_dynamic_field) {
    enable_dynamic_field_ = enable_dynamic_field;
}

CreateSimpleCollectionRequest&
CreateSimpleCollectionRequest::WithEnableDynamicField(bool enable_dynamic_field) {
    SetEnableDynamicField(enable_dynamic_field);
    return *this;
}

int64_t
CreateSimpleCollectionRequest::MaxLength() const {
    return max_length_;
}

void
CreateSimpleCollectionRequest::SetMaxLength(int64_t max_length) {
    max_length_ = max_length;
}

CreateSimpleCollectionRequest&
CreateSimpleCollectionRequest::WithMaxLength(int64_t max_length) {
    SetMaxLength(max_length);
    return *this;
}

}  // namespace milvus