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

#include "milvus/types/SubSearchRequest.h"

#include <memory>

namespace milvus {

SubSearchRequest&
SubSearchRequest::WithMetricType(::milvus::MetricType metric_type) {
    SetMetricType(metric_type);
    return *this;
}

SubSearchRequest&
SubSearchRequest::WithLimit(int64_t limit) {
    SetLimit(limit);
    return *this;
}

SubSearchRequest&
SubSearchRequest::WithFilter(std::string filter) {
    SetFilter(std::move(filter));
    return *this;
}

SubSearchRequest&
SubSearchRequest::WithAnnsField(const std::string& ann_field) {
    SetAnnsField(ann_field);
    return *this;
}

SubSearchRequest&
SubSearchRequest::AddBinaryVector(const std::string& vector) {
    SearchRequestBase::AddBinaryVector(vector);
    return *this;
}

SubSearchRequest&
SubSearchRequest::AddBinaryVector(const BinaryVecFieldData::ElementT& vector) {
    SearchRequestBase::AddBinaryVector(vector);
    return *this;
}

SubSearchRequest&
SubSearchRequest::AddFloatVector(const FloatVecFieldData::ElementT& vector) {
    SearchRequestBase::AddFloatVector(vector);
    return *this;
}

SubSearchRequest&
SubSearchRequest::AddSparseVector(const SparseFloatVecFieldData::ElementT& vector) {
    SearchRequestBase::AddSparseVector(vector);
    return *this;
}

SubSearchRequest&
SubSearchRequest::AddSparseVector(const nlohmann::json& vector) {
    SearchRequestBase::AddSparseVector(vector);
    return *this;
}

SubSearchRequest&
SubSearchRequest::AddFloat16Vector(const Float16VecFieldData::ElementT& vector) {
    SearchRequestBase::AddFloat16Vector(vector);
    return *this;
}

SubSearchRequest&
SubSearchRequest::AddFloat16Vector(const std::vector<float>& vector) {
    SearchRequestBase::AddFloat16Vector(vector);
    return *this;
}

SubSearchRequest&
SubSearchRequest::AddBFloat16Vector(const BFloat16VecFieldData::ElementT& vector) {
    SearchRequestBase::AddBFloat16Vector(vector);
    return *this;
}

SubSearchRequest&
SubSearchRequest::AddBFloat16Vector(const std::vector<float>& vector) {
    SearchRequestBase::AddBFloat16Vector(vector);
    return *this;
}

SubSearchRequest&
SubSearchRequest::AddEmbeddedText(const std::string& text) {
    SearchRequestBase::AddEmbeddedText(text);
    return *this;
}

SubSearchRequest&
SubSearchRequest::AddInt8Vector(const Int8VecFieldData::ElementT& vector) {
    SearchRequestBase::AddInt8Vector(vector);
    return *this;
}

SubSearchRequest&
SubSearchRequest::AddEmbeddingList(EmbeddingList&& emb_list) {
    SearchRequestBase::AddEmbeddingList(std::move(emb_list));
    return *this;
}

}  // namespace milvus
