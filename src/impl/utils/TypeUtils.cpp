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

#include "TypeUtils.h"

#include "./Constants.h"

namespace milvus {

proto::schema::DataType
DataTypeCast(DataType type) {
    switch (type) {
        case DataType::BOOL:
            return proto::schema::DataType::Bool;
        case DataType::INT8:
            return proto::schema::DataType::Int8;
        case DataType::INT16:
            return proto::schema::DataType::Int16;
        case DataType::INT32:
            return proto::schema::DataType::Int32;
        case DataType::INT64:
            return proto::schema::DataType::Int64;
        case DataType::FLOAT:
            return proto::schema::DataType::Float;
        case DataType::DOUBLE:
            return proto::schema::DataType::Double;
        case DataType::VARCHAR:
            return proto::schema::DataType::VarChar;
        case DataType::JSON:
            return proto::schema::DataType::JSON;
        case DataType::ARRAY:
            return proto::schema::DataType::Array;
        case DataType::BINARY_VECTOR:
            return proto::schema::DataType::BinaryVector;
        case DataType::FLOAT_VECTOR:
            return proto::schema::DataType::FloatVector;
        case DataType::SPARSE_FLOAT_VECTOR:
            return proto::schema::DataType::SparseFloatVector;
        case DataType::FLOAT16_VECTOR:
            return proto::schema::DataType::Float16Vector;
        case DataType::BFLOAT16_VECTOR:
            return proto::schema::DataType::BFloat16Vector;
        case DataType::INT8_VECTOR:
            return proto::schema::DataType::Int8Vector;
        default:
            return proto::schema::DataType::None;
    }
}

DataType
DataTypeCast(proto::schema::DataType type) {
    switch (type) {
        case proto::schema::DataType::Bool:
            return DataType::BOOL;
        case proto::schema::DataType::Int8:
            return DataType::INT8;
        case proto::schema::DataType::Int16:
            return DataType::INT16;
        case proto::schema::DataType::Int32:
            return DataType::INT32;
        case proto::schema::DataType::Int64:
            return DataType::INT64;
        case proto::schema::DataType::Float:
            return DataType::FLOAT;
        case proto::schema::DataType::Double:
            return DataType::DOUBLE;
        case proto::schema::DataType::VarChar:
            return DataType::VARCHAR;
        case proto::schema::DataType::JSON:
            return DataType::JSON;
        case proto::schema::DataType::Array:
            return DataType::ARRAY;
        case proto::schema::DataType::BinaryVector:
            return DataType::BINARY_VECTOR;
        case proto::schema::DataType::FloatVector:
            return DataType::FLOAT_VECTOR;
        case proto::schema::DataType::SparseFloatVector:
            return DataType::SPARSE_FLOAT_VECTOR;
        case proto::schema::DataType::Float16Vector:
            return DataType::FLOAT16_VECTOR;
        case proto::schema::DataType::BFloat16Vector:
            return DataType::BFLOAT16_VECTOR;
        case proto::schema::DataType::Int8Vector:
            return DataType::INT8_VECTOR;
        default:
            return DataType::UNKNOWN;
    }
}

proto::schema::FunctionType
FunctionTypeCast(FunctionType type) {
    switch (type) {
        case FunctionType::BM25:
            return proto::schema::FunctionType::BM25;
        case FunctionType::TEXTEMBEDDING:
            return proto::schema::FunctionType::TextEmbedding;
        case FunctionType::RERANK:
            return proto::schema::FunctionType::Rerank;
        default:
            return proto::schema::FunctionType::Unknown;
    }
}

FunctionType
FunctionTypeCast(proto::schema::FunctionType type) {
    switch (type) {
        case proto::schema::FunctionType::BM25:
            return FunctionType::BM25;
        case proto::schema::FunctionType::TextEmbedding:
            return FunctionType::TEXTEMBEDDING;
        case proto::schema::FunctionType::Rerank:
            return FunctionType::RERANK;
        default:
            return FunctionType::UNKNOWN;
    }
}

MetricType
MetricTypeCast(const std::string& type) {
    if (type == "L2") {
        return MetricType::L2;
    }
    if (type == "IP") {
        return MetricType::IP;
    }
    if (type == "COSINE") {
        return MetricType::COSINE;
    }
    if (type == "HAMMING") {
        return MetricType::HAMMING;
    }
    if (type == "JACCARD") {
        return MetricType::JACCARD;
    }
    if (type == "MHJACCARD") {
        return MetricType::MHJACCARD;
    }
    if (type == "BM25") {
        return MetricType::BM25;
    }
    if (type == "MAX_SIM_COSINE") {
        return MetricType::MAX_SIM_COSINE;
    }
    if (type == "MAX_SIM_IP") {
        return MetricType::MAX_SIM_IP;
    }
    if (type == "MAX_SIM_L2") {
        return MetricType::MAX_SIM_L2;
    }
    if (type == "MAX_SIM_JACCARD") {
        return MetricType::MAX_SIM_JACCARD;
    }
    if (type == "MAX_SIM_HAMMING") {
        return MetricType::MAX_SIM_HAMMING;
    }
    return MetricType::DEFAULT;
}

IndexType
IndexTypeCast(const std::string& type) {
    if (type == "FLAT") {
        return IndexType::FLAT;
    }
    if (type == "IVF_FLAT") {
        return IndexType::IVF_FLAT;
    }
    if (type == "IVF_SQ8") {
        return IndexType::IVF_SQ8;
    }
    if (type == "IVF_PQ") {
        return IndexType::IVF_PQ;
    }
    if (type == "HNSW") {
        return IndexType::HNSW;
    }
    if (type == "HNSW_SQ") {
        return IndexType::HNSW_SQ;
    }
    if (type == "HNSW_PQ") {
        return IndexType::HNSW_PQ;
    }
    if (type == "HNSW_PRQ") {
        return IndexType::HNSW_PRQ;
    }
    if (type == "DISKANN") {
        return IndexType::DISKANN;
    }
    if (type == "AUTOINDEX") {
        return IndexType::AUTOINDEX;
    }
    if (type == "SCANN") {
        return IndexType::SCANN;
    }
    if (type == "IVF_RABITQ") {
        return IndexType::IVF_RABITQ;
    }
    if (type == "GPU_IVF_FLAT") {
        return IndexType::GPU_IVF_FLAT;
    }
    if (type == "GPU_IVF_PQ") {
        return IndexType::GPU_IVF_PQ;
    }
    if (type == "GPU_BRUTE_FORCE") {
        return IndexType::GPU_BRUTE_FORCE;
    }
    if (type == "GPU_CAGRA") {
        return IndexType::GPU_CAGRA;
    }
    if (type == "BIN_FLAT") {
        return IndexType::BIN_FLAT;
    }
    if (type == "BIN_IVF_FLAT") {
        return IndexType::BIN_IVF_FLAT;
    }
    if (type == "MINHASH_LSH") {
        return IndexType::MINHASH_LSH;
    }
    if (type == "Trie") {
        return IndexType::TRIE;
    }
    if (type == "STL_SORT") {
        return IndexType::STL_SORT;
    }
    if (type == "INVERTED") {
        return IndexType::INVERTED;
    }
    if (type == "BITMAP") {
        return IndexType::BITMAP;
    }
    if (type == "NGRAM") {
        return IndexType::NGRAM;
    }
    if (type == "SPARSE_INVERTED_INDEX") {
        return IndexType::SPARSE_INVERTED_INDEX;
    }
    if (type == "SPARSE_WAND") {
        return IndexType::SPARSE_WAND;
    }
    return IndexType::INVALID;
}

LoadState
LoadStateCast(proto::common::LoadState state) {
    switch (state) {
        case proto::common::LoadState::LoadStateNotExist:
            return LoadState::LOAD_STATE_NOT_EXIST;
        case proto::common::LoadState::LoadStateNotLoad:
            return LoadState::LOAD_STATE_NOT_LOAD;
        case proto::common::LoadState::LoadStateLoading:
            return LoadState::LOAD_STATE_LOADING;
        case proto::common::LoadState::LoadStateLoaded:
            return LoadState::LOAD_STATE_LOADED;
        default:
            return LoadState::LOAD_STATE_NOT_EXIST;
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// methods for schema types converting
void
ConvertValueFieldSchema(const proto::schema::ValueField& value_field, DataType type, nlohmann::json& val) {
    switch (type) {
        case DataType::BOOL:
            if (value_field.has_bool_data()) {
                val = value_field.bool_data();
            }
            return;
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32:
            if (value_field.has_int_data()) {
                val = value_field.int_data();
            }
            return;
        case DataType::INT64:
            if (value_field.has_long_data()) {
                val = value_field.long_data();
            }
            return;
        case DataType::FLOAT:
            if (value_field.has_float_data()) {
                val = value_field.float_data();
            }
            return;
        case DataType::DOUBLE:
            if (value_field.has_double_data()) {
                val = value_field.double_data();
            }
            return;
        case DataType::VARCHAR:
            if (value_field.has_string_data()) {
                val = value_field.string_data();
            }
            return;
        case DataType::JSON:
            if (value_field.has_string_data()) {
                val = nlohmann::json::parse(value_field.string_data());
            }
            return;
        default:
            return;
    }
}

void
ConvertFieldSchema(const proto::schema::FieldSchema& proto_schema, FieldSchema& field_schema) {
    field_schema.SetName(proto_schema.name());
    field_schema.SetDescription(proto_schema.description());
    field_schema.SetPrimaryKey(proto_schema.is_primary_key());
    field_schema.SetPartitionKey(proto_schema.is_partition_key());
    field_schema.SetClusteringKey(proto_schema.is_clustering_key());
    field_schema.SetAutoID(proto_schema.autoid());
    DataType type = DataTypeCast(proto_schema.data_type());
    field_schema.SetDataType(type);
    field_schema.SetElementType(DataTypeCast(proto_schema.element_type()));
    field_schema.SetNullable(proto_schema.nullable());

    nlohmann::json default_value;
    ConvertValueFieldSchema(proto_schema.default_value(), type, default_value);
    if (!default_value.is_null()) {
        field_schema.SetDefaultValue(default_value);
    }

    std::map<std::string, std::string> params;
    for (int k = 0; k < proto_schema.type_params_size(); ++k) {
        auto& kv = proto_schema.type_params(k);
        params.emplace(kv.key(), kv.value());
    }
    field_schema.SetTypeParams(std::move(params));
}

void
ConvertFunctionSchema(const proto::schema::FunctionSchema& proto_function, FunctionPtr& function_schema) {
    function_schema = std::make_shared<Function>();
    function_schema->SetName(proto_function.name());
    function_schema->SetDescription(proto_function.description());
    function_schema->SetFunctionType(FunctionTypeCast(proto_function.type()));

    for (const auto& name : proto_function.input_field_names()) {
        function_schema->AddInputFieldName(name);
    }
    for (const auto& name : proto_function.output_field_names()) {
        function_schema->AddOutputFieldName(name);
    }

    for (const auto& kv : proto_function.params()) {
        function_schema->AddParam(kv.key(), kv.value());
    }
}

void
ConvertCollectionSchema(const proto::schema::CollectionSchema& proto_schema, CollectionSchema& schema) {
    schema.SetName(proto_schema.name());
    schema.SetDescription(proto_schema.description());
    schema.SetEnableDynamicField(proto_schema.enable_dynamic_field());

    for (int i = 0; i < proto_schema.fields_size(); ++i) {
        auto& proto_field = proto_schema.fields(i);
        FieldSchema field_schema;
        ConvertFieldSchema(proto_field, field_schema);
        schema.AddField(std::move(field_schema));
    }

    for (int i = 0; i < proto_schema.functions_size(); ++i) {
        auto& proto_function = proto_schema.functions(i);
        FunctionPtr function_schema;
        ConvertFunctionSchema(proto_function, function_schema);
        schema.AddFunction(function_schema);
    }
}

Status
CheckDefaultValue(const FieldSchema& schema) {
    const nlohmann::json& val = schema.DefaultValue();
    if (val.is_null()) {
        // default value is not set, return ok
        return Status::OK();
    }

    DataType type = schema.FieldDataType();
    switch (type) {
        case DataType::BOOL:
            if (!val.is_boolean()) {
                return {StatusCode::INVALID_AGUMENT, "Field type is BOOL but default value is not boolean"};
            }
            break;
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::INT64:
            // TODO: check ranges?
            if (!val.is_number_integer() && !val.is_number_unsigned()) {
                return {StatusCode::INVALID_AGUMENT, "Field type is INT but default value is not integer"};
            }
            break;
        case DataType::FLOAT:
        case DataType::DOUBLE:
            if (!val.is_number()) {
                return {StatusCode::INVALID_AGUMENT, "Field type is FLOAT/DOUBLE but default value is not number"};
            }
            break;
        case DataType::VARCHAR:
            if (!val.is_string()) {
                return {StatusCode::INVALID_AGUMENT, "Field type is VARCHAR but default value is not string"};
            }
            break;
        case DataType::JSON:
            return Status::OK();
        default:
            return {StatusCode::INVALID_AGUMENT, "Not allow to set default value for " + std::to_string(type)};
    }

    return Status::OK();
}

void
ConvertValueFieldSchema(const nlohmann::json& val, DataType type, proto::schema::ValueField& value_field) {
    // we suppose the val has been verified by CheckDefaultValue
    if (val.is_null()) {
        return;
    }
    switch (type) {
        case DataType::BOOL:
            value_field.set_bool_data(val.get<bool>());
            return;
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32:
            value_field.set_int_data(val.get<int32_t>());
            return;
        case DataType::INT64:
            value_field.set_long_data(val.get<int64_t>());
            return;
        case DataType::FLOAT:
            value_field.set_float_data(val.get<float>());
            return;
        case DataType::DOUBLE:
            value_field.set_double_data(val.get<double>());
            return;
        case DataType::VARCHAR:
            value_field.set_string_data(val.get<std::string>());
            return;
        case DataType::JSON:
            value_field.set_string_data(val.dump());
            return;
        default:
            return;
    }
}

void
ConvertFieldSchema(const FieldSchema& schema, proto::schema::FieldSchema& proto_schema) {
    proto_schema.set_name(schema.Name());
    proto_schema.set_description(schema.Description());
    proto_schema.set_is_primary_key(schema.IsPrimaryKey());
    proto_schema.set_is_partition_key(schema.IsPartitionKey());
    proto_schema.set_is_clustering_key(schema.IsClusteringKey());
    proto_schema.set_autoid(schema.AutoID());
    proto_schema.set_data_type(DataTypeCast(schema.FieldDataType()));
    proto_schema.set_nullable(schema.IsNullable());

    if (!schema.DefaultValue().is_null()) {
        ConvertValueFieldSchema(schema.DefaultValue(), schema.FieldDataType(), *proto_schema.mutable_default_value());
    }

    if (schema.FieldDataType() == DataType::ARRAY) {
        proto_schema.set_element_type(DataTypeCast(schema.ElementType()));
    }

    for (auto& kv : schema.TypeParams()) {
        auto pair = proto_schema.add_type_params();
        pair->set_key(kv.first);
        pair->set_value(kv.second);
    }
}

void
ConvertFunctionSchema(const FunctionPtr& function_schema, proto::schema::FunctionSchema& proto_function) {
    if (function_schema == nullptr) {
        return;
    }
    proto_function.set_name(function_schema->Name());
    proto_function.set_description(function_schema->Description());
    proto_function.set_type(FunctionTypeCast(function_schema->GetFunctionType()));

    for (const auto& input_field : function_schema->InputFieldNames()) {
        proto_function.add_input_field_names(input_field);
    }
    for (const auto& output_field : function_schema->OutputFieldNames()) {
        proto_function.add_output_field_names(output_field);
    }

    for (const auto& pair : function_schema->Params()) {
        auto kv = proto_function.add_params();
        kv->set_key(pair.first);
        kv->set_value(pair.second);
    }
}

void
ConvertFunctionScore(const FunctionScorePtr& function_score, proto::schema::FunctionScore& proto_score) {
    if (function_score == nullptr) {
        return;
    }

    const auto& functions = function_score->Functions();
    for (const auto& function : functions) {
        auto proto_function = proto_score.add_functions();
        ConvertFunctionSchema(function, *proto_function);
    }

    const auto& params = function_score->Params();
    for (const auto& param : params) {
        auto kv = proto_score.add_params();
        kv->set_key(param.first);
        kv->set_value(param.second.dump());
    }
}

void
ConvertCollectionSchema(const CollectionSchema& schema, proto::schema::CollectionSchema& proto_schema) {
    proto_schema.set_name(schema.Name());
    proto_schema.set_description(schema.Description());
    proto_schema.set_enable_dynamic_field(schema.EnableDynamicField());

    for (const auto& field : schema.Fields()) {
        auto proto_field = proto_schema.add_fields();
        ConvertFieldSchema(field, *proto_field);
    }

    for (const auto& function : schema.Functions()) {
        auto proto_function = proto_schema.add_functions();
        ConvertFunctionSchema(function, *proto_function);
    }
}

SegmentState
SegmentStateCast(proto::common::SegmentState state) {
    switch (state) {
        case proto::common::SegmentState::Dropped:
            return SegmentState::DROPPED;
        case proto::common::SegmentState::Flushed:
            return SegmentState::FLUSHED;
        case proto::common::SegmentState::Flushing:
            return SegmentState::FLUSHING;
        case proto::common::SegmentState::Growing:
            return SegmentState::GROWING;
        case proto::common::SegmentState::NotExist:
            return SegmentState::NOT_EXIST;
        case proto::common::SegmentState::Sealed:
            return SegmentState::SEALED;
        default:
            return SegmentState::UNKNOWN;
    }
}

proto::common::SegmentState
SegmentStateCast(SegmentState state) {
    switch (state) {
        case SegmentState::DROPPED:
            return proto::common::SegmentState::Dropped;
        case SegmentState::FLUSHED:
            return proto::common::SegmentState::Flushed;
        case SegmentState::FLUSHING:
            return proto::common::SegmentState::Flushing;
        case SegmentState::GROWING:
            return proto::common::SegmentState::Growing;
        case SegmentState::NOT_EXIST:
            return proto::common::SegmentState::NotExist;
        case SegmentState::SEALED:
            return proto::common::SegmentState::Sealed;
        default:
            return proto::common::SegmentState::SegmentStateNone;
    }
}

IndexStateCode
IndexStateCast(proto::common::IndexState state) {
    switch (state) {
        case proto::common::IndexState::IndexStateNone:
            return IndexStateCode::NONE;
        case proto::common::IndexState::Unissued:
            return IndexStateCode::UNISSUED;
        case proto::common::IndexState::InProgress:
            return IndexStateCode::IN_PROGRESS;
        case proto::common::IndexState::Finished:
            return IndexStateCode::FINISHED;
        default:
            return IndexStateCode::FAILED;
    }
}

bool
IsVectorType(DataType type) {
    return (DataType::BINARY_VECTOR == type || DataType::FLOAT_VECTOR == type ||
            DataType::SPARSE_FLOAT_VECTOR == type || DataType::FLOAT16_VECTOR == type ||
            DataType::BFLOAT16_VECTOR == type || DataType::INT8_VECTOR == type);
}

std::string
Base64Encode(const std::string& val) {
    const char* base64_chars = {
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        "+/"};

    auto len = val.size();
    auto len_encoded = (len + 2) / 3 * 4;
    std::string ret;
    ret.reserve(len_encoded);

    size_t pos = 0;

    while (pos < len) {
        ret.push_back(base64_chars[(val[pos + 0] & 0xfc) >> 2]);

        if (pos + 1 < len) {
            ret.push_back(base64_chars[((val[pos + 0] & 0x03) << 4) + ((val[pos + 1] & 0xf0) >> 4)]);

            if (pos + 2 < len) {
                ret.push_back(base64_chars[((val[pos + 1] & 0x0f) << 2) + ((val[pos + 2] & 0xc0) >> 6)]);
                ret.push_back(base64_chars[val[pos + 2] & 0x3f]);
            } else {
                ret.push_back(base64_chars[(val[pos + 1] & 0x0f) << 2]);
                ret.push_back('=');
            }
        } else {
            ret.push_back(base64_chars[(val[pos + 0] & 0x03) << 4]);
            ret.push_back('=');
            ret.push_back('=');
        }

        pos += 3;
    }

    return ret;
}

proto::common::ConsistencyLevel
ConsistencyLevelCast(const ConsistencyLevel& level) {
    switch (level) {
        case ConsistencyLevel::STRONG:
            return proto::common::ConsistencyLevel::Strong;
        case ConsistencyLevel::SESSION:
            return proto::common::ConsistencyLevel::Session;
        case ConsistencyLevel::EVENTUALLY:
            return proto::common::ConsistencyLevel::Eventually;
        default:
            return proto::common::ConsistencyLevel::Bounded;
    }
}

ConsistencyLevel
ConsistencyLevelCast(const proto::common::ConsistencyLevel& level) {
    switch (level) {
        case proto::common::ConsistencyLevel::Strong:
            return ConsistencyLevel::STRONG;
        case proto::common::ConsistencyLevel::Session:
            return ConsistencyLevel::SESSION;
        case proto::common::ConsistencyLevel::Eventually:
            return ConsistencyLevel::EVENTUALLY;
        default:
            return ConsistencyLevel::BOUNDED;
    }
}

void
ConvertResourceGroupConfig(const ResourceGroupConfig& config, proto::rg::ResourceGroupConfig* rpc_config) {
    rpc_config->mutable_requests()->set_node_num(static_cast<int32_t>(config.Requests()));
    rpc_config->mutable_limits()->set_node_num(static_cast<int32_t>(config.Limits()));

    for (const auto& name : config.TransferFromGroups()) {
        proto::rg::ResourceGroupTransfer transfer;
        transfer.set_resource_group(name);
        rpc_config->mutable_transfer_from()->Add(std::move(transfer));
    }
    for (const auto& name : config.TransferToGroups()) {
        proto::rg::ResourceGroupTransfer transfer;
        transfer.set_resource_group(name);
        rpc_config->mutable_transfer_to()->Add(std::move(transfer));
    }
    for (const auto& pair : config.NodeFilters()) {
        auto kv = rpc_config->mutable_node_filter()->add_node_labels();
        kv->set_key(pair.first);
        kv->set_value(pair.second);
    }
}

void
ConvertResourceGroupConfig(const proto::rg::ResourceGroupConfig& rpc_config, ResourceGroupConfig& config) {
    config.SetRequests(static_cast<uint32_t>(rpc_config.requests().node_num()));
    config.SetLimits(static_cast<uint32_t>(rpc_config.limits().node_num()));

    for (const auto& transfer : rpc_config.transfer_from()) {
        config.AddTrnasferFromGroup(transfer.resource_group());
    }
    for (const auto& transfer : rpc_config.transfer_to()) {
        config.AddTrnasferToGroup(transfer.resource_group());
    }
    for (const auto& kv : rpc_config.node_filter().node_labels()) {
        config.AddNodeFilter(kv.key(), kv.value());
    }
}

bool
IsValidTemplate(const nlohmann::json& filter_template) {
    return filter_template.is_boolean() || filter_template.is_number() || filter_template.is_string();
}

}  // namespace milvus

namespace std {
std::string
to_string(milvus::MetricType metric_type) {
    switch (metric_type) {
        case milvus::MetricType::L2:
            return "L2";
        case milvus::MetricType::IP:
            return "IP";
        case milvus::MetricType::COSINE:
            return "COSINE";
        case milvus::MetricType::HAMMING:
            return "HAMMING";
        case milvus::MetricType::JACCARD:
            return "JACCARD";
        case milvus::MetricType::MHJACCARD:
            return "MHJACCARD";
        case milvus::MetricType::BM25:
            return "BM25";
        case milvus::MetricType::MAX_SIM_COSINE:
            return "MAX_SIM_COSINE";
        case milvus::MetricType::MAX_SIM_IP:
            return "MAX_SIM_IP";
        case milvus::MetricType::MAX_SIM_L2:
            return "MAX_SIM_L2";
        case milvus::MetricType::MAX_SIM_JACCARD:
            return "MAX_SIM_JACCARD";
        case milvus::MetricType::MAX_SIM_HAMMING:
            return "MAX_SIM_HAMMING";
        default:
            return "DEFAULT";
    }
}
std::string
to_string(milvus::IndexType index_type) {
    switch (index_type) {
        case milvus::IndexType::FLAT:
            return "FLAT";
        case milvus::IndexType::IVF_FLAT:
            return "IVF_FLAT";
        case milvus::IndexType::IVF_PQ:
            return "IVF_PQ";
        case milvus::IndexType::IVF_SQ8:
            return "IVF_SQ8";
        case milvus::IndexType::HNSW:
            return "HNSW";
        case milvus::IndexType::HNSW_SQ:
            return "HNSW_SQ";
        case milvus::IndexType::HNSW_PQ:
            return "HNSW_PQ";
        case milvus::IndexType::HNSW_PRQ:
            return "HNSW_PRQ";
        case milvus::IndexType::DISKANN:
            return "DISKANN";
        case milvus::IndexType::AUTOINDEX:
            return "AUTOINDEX";
        case milvus::IndexType::SCANN:
            return "SCANN";
        case milvus::IndexType::IVF_RABITQ:
            return "IVF_RABITQ";
        case milvus::IndexType::GPU_IVF_FLAT:
            return "GPU_IVF_FLAT";
        case milvus::IndexType::GPU_IVF_PQ:
            return "GPU_IVF_PQ";
        case milvus::IndexType::GPU_BRUTE_FORCE:
            return "GPU_BRUTE_FORCE";
        case milvus::IndexType::GPU_CAGRA:
            return "GPU_CAGRA";
        case milvus::IndexType::BIN_FLAT:
            return "BIN_FLAT";
        case milvus::IndexType::BIN_IVF_FLAT:
            return "BIN_IVF_FLAT";
        case milvus::IndexType::MINHASH_LSH:
            return "MINHASH_LSH";
        case milvus::IndexType::TRIE:
            return "Trie";
        case milvus::IndexType::STL_SORT:
            return "STL_SORT";
        case milvus::IndexType::INVERTED:
            return "INVERTED";
        case milvus::IndexType::BITMAP:
            return "BITMAP";
        case milvus::IndexType::NGRAM:
            return "NGRAM";
        case milvus::IndexType::SPARSE_INVERTED_INDEX:
            return "SPARSE_INVERTED_INDEX";
        case milvus::IndexType::SPARSE_WAND:
            return "SPARSE_WAND";
        default:
            return "INVALID";
    }
}

std::string
to_string(milvus::DataType data_type) {
    static const std::map<milvus::DataType, std::string> name_map = {
        {milvus::DataType::BOOL, "BOOL"},
        {milvus::DataType::INT8, "INT8"},
        {milvus::DataType::INT16, "INT8"},
        {milvus::DataType::INT32, "INT32"},
        {milvus::DataType::INT64, "INT64"},
        {milvus::DataType::FLOAT, "FLOAT"},
        {milvus::DataType::DOUBLE, "DOUBLE"},
        {milvus::DataType::VARCHAR, "VARCHAR"},
        {milvus::DataType::JSON, "JSON"},
        {milvus::DataType::ARRAY, "ARRAY"},
        {milvus::DataType::BINARY_VECTOR, "BINARY_VECTOR"},
        {milvus::DataType::FLOAT_VECTOR, "FLOAT_VECTOR"},
        {milvus::DataType::FLOAT16_VECTOR, "FLOAT16_VECTOR"},
        {milvus::DataType::BFLOAT16_VECTOR, "BFLOAT16_VECTOR"},
        {milvus::DataType::SPARSE_FLOAT_VECTOR, "SPARSE_FLOAT_VECTOR"},
        {milvus::DataType::INT8_VECTOR, "INT8_VECTOR"},
    };
    auto it = name_map.find(data_type);
    if (it == name_map.end()) {
        return "Unknow DataType";
    }
    return it->second;
}

}  // namespace std
