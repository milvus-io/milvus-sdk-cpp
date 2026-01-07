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

#include "milvus/types/EmbeddingList.h"

#include "../utils/DmlUtils.h"
#include "milvus/utils/FP16.h"

namespace {

int64_t
deduceDim(milvus::DataType data_type, int64_t length) {
    if (data_type == milvus::DataType::BINARY_VECTOR) {
        return length * 8;
    } else {
        return length;
    }
}

std::vector<uint16_t>
toVector16(const std::vector<float>& vector, bool is_bf16) {
    std::vector<uint16_t> binary;
    binary.reserve(vector.size());
    for (auto val : vector) {
        binary.push_back(is_bf16 ? milvus::F32toBF16(val) : milvus::F32toF16(val));
    }
    return binary;
}

}  // namespace

namespace milvus {

FieldDataPtr
EmbeddingList::TargetVectors() const {
    return target_vectors_;
}

size_t
EmbeddingList::Count() const {
    return target_vectors_ == nullptr ? 0 : target_vectors_->Count();
}

int64_t
EmbeddingList::Dim() const {
    return dim_;
}

////////////////////////////////////////////////////////////////////////
// single vector assigner
Status
EmbeddingList::AddBinaryVector(const std::string& vector) {
    return AddBinaryVector(BinaryVecFieldData::ToUnsignedChars(vector));
}

Status
EmbeddingList::AddBinaryVector(const BinaryVecFieldData::ElementT& vector) {
    return addVector<BinaryVecFieldData, BinaryVecFieldData::ElementT>(DataType::BINARY_VECTOR, vector);
}

Status
EmbeddingList::AddFloatVector(const FloatVecFieldData::ElementT& vector) {
    return addVector<FloatVecFieldData, FloatVecFieldData::ElementT>(DataType::FLOAT_VECTOR, vector);
}

Status
EmbeddingList::AddSparseVector(const SparseFloatVecFieldData::ElementT& vector) {
    return addVector<SparseFloatVecFieldData, SparseFloatVecFieldData::ElementT>(DataType::SPARSE_FLOAT_VECTOR, vector);
}

Status
EmbeddingList::AddSparseVector(const nlohmann::json& vector) {
    std::map<uint32_t, float> pairs;
    auto status = ParseSparseFloatVector(vector, "", pairs);
    if (!status.IsOk()) {
        return status;
    }
    return AddSparseVector(pairs);
}

Status
EmbeddingList::AddFloat16Vector(const Float16VecFieldData::ElementT& vector) {
    return addVector<Float16VecFieldData, Float16VecFieldData::ElementT>(DataType::FLOAT16_VECTOR, vector);
}

Status
EmbeddingList::AddFloat16Vector(const std::vector<float>& vector) {
    std::vector<uint16_t> binary = toVector16(vector, false);
    return AddFloat16Vector(binary);
}

Status
EmbeddingList::AddBFloat16Vector(const BFloat16VecFieldData::ElementT& vector) {
    return addVector<BFloat16VecFieldData, BFloat16VecFieldData::ElementT>(DataType::BFLOAT16_VECTOR, vector);
}

Status
EmbeddingList::AddBFloat16Vector(const std::vector<float>& vector) {
    std::vector<uint16_t> binary = toVector16(vector, true);
    return AddBFloat16Vector(binary);
}

Status
EmbeddingList::AddEmbeddedText(const std::string& text) {
    return addVector<VarCharFieldData, VarCharFieldData::ElementT>(DataType::VARCHAR, text);
}

Status
EmbeddingList::AddInt8Vector(const Int8VecFieldData::ElementT& vector) {
    return addVector<Int8VecFieldData, Int8VecFieldData::ElementT>(DataType::INT8_VECTOR, vector);
}

////////////////////////////////////////////////////////////////////////
// multi vectors assigner
Status
EmbeddingList::SetBinaryVectors(const std::vector<std::string>& vectors) {
    std::vector<BinaryVecFieldData::ElementT> actual_vectors;
    actual_vectors.reserve(vectors.size());
    for (const auto& vector : vectors) {
        actual_vectors.emplace_back(BinaryVecFieldData::ToUnsignedChars(vector));
    }

    return SetBinaryVectors(std::move(actual_vectors));
}

Status
EmbeddingList::SetBinaryVectors(std::vector<BinaryVecFieldData::ElementT>&& vectors) {
    return setVectors<BinaryVecFieldData, BinaryVecFieldData::ElementT>(DataType::BINARY_VECTOR, std::move(vectors));
}

Status
EmbeddingList::SetFloatVectors(std::vector<FloatVecFieldData::ElementT>&& vectors) {
    return setVectors<FloatVecFieldData, FloatVecFieldData::ElementT>(DataType::FLOAT_VECTOR, std::move(vectors));
}

Status
EmbeddingList::SetSparseVectors(std::vector<SparseFloatVecFieldData::ElementT>&& vectors) {
    return setVectors<SparseFloatVecFieldData, SparseFloatVecFieldData::ElementT>(DataType::SPARSE_FLOAT_VECTOR,
                                                                                  std::move(vectors));
}

Status
EmbeddingList::SetSparseVectors(const std::vector<nlohmann::json>& vectors) {
    std::vector<SparseFloatVecFieldData::ElementT> actual_vectors;
    actual_vectors.reserve(vectors.size());
    for (const auto& vector : vectors) {
        std::map<uint32_t, float> pairs;
        auto status = ParseSparseFloatVector(vector, "", pairs);
        if (!status.IsOk()) {
            return status;
        }
        actual_vectors.emplace_back(pairs);
    }

    return SetSparseVectors(std::move(actual_vectors));
}

Status
EmbeddingList::SetFloat16Vectors(std::vector<Float16VecFieldData::ElementT>&& vectors) {
    return setVectors<Float16VecFieldData, Float16VecFieldData::ElementT>(DataType::FLOAT16_VECTOR, std::move(vectors));
}

Status
EmbeddingList::SetFloat16Vectors(const std::vector<std::vector<float>>& vectors) {
    std::vector<Float16VecFieldData::ElementT> actual_vectors;
    actual_vectors.reserve(vectors.size());
    for (const auto& vector : vectors) {
        std::vector<uint16_t> binary = toVector16(vector, false);
        actual_vectors.emplace_back(binary);
    }

    return SetFloat16Vectors(std::move(actual_vectors));
}

Status
EmbeddingList::SetBFloat16Vectors(std::vector<BFloat16VecFieldData::ElementT>&& vectors) {
    return setVectors<BFloat16VecFieldData, BFloat16VecFieldData::ElementT>(DataType::BFLOAT16_VECTOR,
                                                                            std::move(vectors));
}

Status
EmbeddingList::SetBFloat16Vectors(const std::vector<std::vector<float>>& vectors) {
    std::vector<BFloat16VecFieldData::ElementT> actual_vectors;
    actual_vectors.reserve(vectors.size());
    for (const auto& vector : vectors) {
        std::vector<uint16_t> binary = toVector16(vector, true);
        actual_vectors.emplace_back(binary);
    }

    return SetBFloat16Vectors(std::move(actual_vectors));
}

Status
EmbeddingList::SetEmbeddedTexts(std::vector<std::string>&& texts) {
    return setVectors<VarCharFieldData, VarCharFieldData::ElementT>(DataType::VARCHAR, std::move(texts));
}

Status
EmbeddingList::SetInt8Vectors(std::vector<Int8VecFieldData::ElementT>&& vectors) {
    return setVectors<Int8VecFieldData, Int8VecFieldData::ElementT>(DataType::INT8_VECTOR, std::move(vectors));
}

//////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, typename V>
Status
EmbeddingList::addVector(DataType data_type, const V& vector) {
    if (target_vectors_ != nullptr && target_vectors_->Type() != data_type) {
        return {StatusCode::INVALID_AGUMENT, "Target vector must be the same type!"};
    }

    StatusCode code = StatusCode::OK;
    auto dim = deduceDim(data_type, static_cast<int64_t>(vector.size()));
    if (nullptr == target_vectors_) {
        std::shared_ptr<T> vectors = std::make_shared<T>("");
        target_vectors_ = vectors;
        code = vectors->Add(vector);
        dim_ = dim;
    } else {
        // dimensions must be equal, except text and sparse
        if (data_type != DataType::VARCHAR && data_type != DataType::SPARSE_FLOAT_VECTOR && dim_ != dim) {
            std::string msg =
                "Vector size mismatch, first: " + std::to_string(dim_) + ", current: " + std::to_string(dim);
            return {StatusCode::INVALID_AGUMENT, msg};
        }

        std::shared_ptr<T> vectors = std::static_pointer_cast<T>(target_vectors_);
        code = vectors->Add(vector);
    }

    if (code != StatusCode::OK) {
        return {code, "Failed to add " + std::to_string(data_type)};
    }

    return Status::OK();
}

template <typename T, typename V>
Status
EmbeddingList::setVectors(DataType data_type, std::vector<V>&& vectors) {
    if (vectors.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Vector list is empty"};
    }

    // this method will reset the vector list
    // there is a risk that the dimension of vectors is different, the returned status is ignored by
    // SearchRequest::WithXXXVectors() hence the illegal vectors are passed to the server, and the server
    // returns an error to the client.
    dim_ = deduceDim(data_type, static_cast<int64_t>(vectors.at(0).size()));
    Status status = Status::OK();
    for (auto& vector : vectors) {
        // dimensions must be equal, except text and sparse
        auto dim = deduceDim(data_type, static_cast<int64_t>(vector.size()));
        if (data_type != DataType::VARCHAR && data_type != DataType::SPARSE_FLOAT_VECTOR && dim_ != dim) {
            std::string msg =
                "Vector size mismatch, first: " + std::to_string(dim_) + ", current: " + std::to_string(dim);
            status = {StatusCode::INVALID_AGUMENT, msg};
        }
    }
    target_vectors_ = std::make_shared<T>("", std::move(vectors));

    return status;
}

}  // namespace milvus
