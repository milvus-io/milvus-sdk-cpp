# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(PROTO_VERSION v2.2.3)
set(PROTO_URL https://github.com/milvus-io/milvus-proto/archive/refs/tags/${PROTO_VERSION}.tar.gz)

include(FetchContent)

# download proto
FetchContent_Declare(milvus_proto
    URL         ${PROTO_URL}
)
FetchContent_Populate(milvus_proto)

find_package(gRPC CONFIG REQUIRED)

set(PROTO_BINARY_DIR "${milvus_proto_BINARY_DIR}")
set(PROTO_IMPORT_DIRS "${milvus_proto_SOURCE_DIR}/proto")

function(add_protos target)
    target_sources(${target} PRIVATE 
        "${milvus_proto_SOURCE_DIR}/proto/common.proto"
        "${milvus_proto_SOURCE_DIR}/proto/schema.proto"
        "${milvus_proto_SOURCE_DIR}/proto/milvus.proto"
    )
    target_include_directories(${target} PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")

    protobuf_generate(
        TARGET ${target}
        IMPORT_DIRS ${PROTO_IMPORT_DIRS}
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")
    
    protobuf_generate(
        TARGET ${target}
        LANGUAGE grpc
        GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
        PLUGIN "protoc-gen-grpc=\$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
        IMPORT_DIRS ${PROTO_IMPORT_DIRS}
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

endfunction(add_protos target)

