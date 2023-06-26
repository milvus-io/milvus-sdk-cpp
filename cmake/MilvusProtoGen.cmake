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

include_guard(GLOBAL)

set(PROTO_VERSION v2.2.11)
set(PROTO_URL https://github.com/milvus-io/milvus-proto/archive/refs/tags/${PROTO_VERSION}.tar.gz)


include(FetchContent)

# download proto
FetchContent_Declare(milvus_proto
    URL         ${PROTO_URL}
)
FetchContent_Populate(milvus_proto)

set(PROTO_BINARY_DIR "${milvus_proto_BINARY_DIR}")
set(PROTO_IMPORT_DIR "${milvus_proto_SOURCE_DIR}/proto")

# resolve protoc, always use the protoc in the build tree
set(Protobuf_PROTOC_EXECUTABLE $<TARGET_FILE:protoc>)
message(STATUS "using protoc: ${Protobuf_PROTOC_EXECUTABLE}")

# resolve grpc_cpp_plugin
set(GRPC_CPP_PLUGIN $<TARGET_FILE:grpc_cpp_plugin>)
message(STATUS "using grpc_cpp_plugin: ${GRPC_CPP_PLUGIN}")


function(add_proto_source target name)
    add_custom_command(
        OUTPUT ${milvus_proto_BINARY_DIR}/${name}.pb.cc
               ${milvus_proto_BINARY_DIR}/${name}.pb.h
        DEPENDS ${PROTO_IMPORT_DIR}/${name}.proto
                ${Protobuf_PROTOC_EXECUTABLE}
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
                --cpp_out ${milvus_proto_BINARY_DIR}
                -I${PROTO_IMPORT_DIR}
                -I${grpc_SOURCE_DIR}/third_party/protobuf/src
                ${PROTO_IMPORT_DIR}/${name}.proto
    )
    target_sources(${target} PRIVATE ${milvus_proto_BINARY_DIR}/${name}.pb.cc)
endfunction(add_proto_source target name)

function(add_proto_service target name)
    add_custom_command(
        OUTPUT ${milvus_proto_BINARY_DIR}/${name}.grpc.pb.cc
               ${milvus_proto_BINARY_DIR}/${name}.grpc.pb.h
        DEPENDS ${PROTO_IMPORT_DIR}/${name}.proto
                ${Protobuf_PROTOC_EXECUTABLE}
                ${GRPC_CPP_PLUGIN}
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
                --grpc_out ${milvus_proto_BINARY_DIR}
                -I${PROTO_IMPORT_DIR}
                -I${grpc_SOURCE_DIR}/third_party/protobuf/src
                --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
                ${PROTO_IMPORT_DIR}/${name}.proto
    )
    target_sources(${target} PRIVATE ${milvus_proto_BINARY_DIR}/${name}.grpc.pb.cc)
endfunction(add_proto_service target name)

function(add_milvus_protos target)
    add_proto_source(${target} "schema")
    add_proto_source(${target} "common")
    add_proto_source(${target} "milvus")
    add_proto_service(${target} "milvus")
    target_include_directories(${target} PRIVATE ${milvus_proto_BINARY_DIR})
endfunction(add_milvus_protos target)

