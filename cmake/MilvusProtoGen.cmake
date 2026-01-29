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

set(PROTO_VERSION v2.6.7)
set(PROTO_URL https://github.com/milvus-io/milvus-proto/archive/refs/tags/${PROTO_VERSION}.tar.gz)


include(FetchContent)

# download proto
FetchContent_Declare(milvus_proto
    URL         ${PROTO_URL}
)
FetchContent_Populate(milvus_proto)

set(PROTO_BINARY_DIR "${milvus_proto_BINARY_DIR}")
set(PROTO_IMPORT_DIR "${milvus_proto_SOURCE_DIR}/proto")

# Milvus protos import Google well-known types (e.g. google/protobuf/descriptor.proto).
# Ensure protoc can find Protobuf's built-in .proto include directory.
set(_milvus_proto_include_args "")

# Prefer include dirs from the imported protobuf target (Conan/package mode).
if (TARGET protobuf::libprotobuf)
    get_target_property(_pb_includes protobuf::libprotobuf INTERFACE_INCLUDE_DIRECTORIES)
    if (_pb_includes)
        foreach(_inc IN LISTS _pb_includes)
            if (_inc)
                list(APPEND _milvus_proto_include_args "-I" "${_inc}")
            endif()
        endforeach()
    endif()
endif()

# Fallback for module/FetchContent builds where protobuf_SOURCE_DIR is set.
if (protobuf_SOURCE_DIR AND EXISTS "${protobuf_SOURCE_DIR}")
    # Common layout for protobuf sources.
    if (EXISTS "${protobuf_SOURCE_DIR}")
        list(APPEND _milvus_proto_include_args "-I" "${protobuf_SOURCE_DIR}")
    endif()
    if (EXISTS "${protobuf_SOURCE_DIR}/src")
        list(APPEND _milvus_proto_include_args "-I" "${protobuf_SOURCE_DIR}/src")
    endif()
endif()

if (NOT TARGET protobuf::protoc)
    message(FATAL_ERROR "protobuf::protoc target not found. Please provide Protobuf via find_package(Protobuf CONFIG REQUIRED).")
endif()

if (NOT TARGET gRPC::grpc_cpp_plugin)
    message(FATAL_ERROR "gRPC::grpc_cpp_plugin target not found. Please provide gRPC via find_package(gRPC CONFIG REQUIRED).")
endif()

set(Protobuf_PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
set(GRPC_CPP_PLUGIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
message(STATUS "using protoc: ${Protobuf_PROTOC_EXECUTABLE}")
message(STATUS "using grpc_cpp_plugin: ${GRPC_CPP_PLUGIN}")


function(add_proto_source target name)
    add_custom_command(
        OUTPUT ${milvus_proto_BINARY_DIR}/${name}.pb.cc
               ${milvus_proto_BINARY_DIR}/${name}.pb.h
        DEPENDS ${PROTO_IMPORT_DIR}/${name}.proto
                ${Protobuf_PROTOC_EXECUTABLE}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        --cpp_out=${milvus_proto_BINARY_DIR}
        -I${PROTO_IMPORT_DIR}
        ${_milvus_proto_include_args}
        ${PROTO_IMPORT_DIR}/${name}.proto
    VERBATIM
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
        --grpc_out=${milvus_proto_BINARY_DIR}
        -I${PROTO_IMPORT_DIR}
        ${_milvus_proto_include_args}
        --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
        ${PROTO_IMPORT_DIR}/${name}.proto
    VERBATIM
    )
    target_sources(${target} PRIVATE ${milvus_proto_BINARY_DIR}/${name}.grpc.pb.cc)
endfunction(add_proto_service target name)

function(add_milvus_protos target)
    add_proto_source(${target} "schema")
    add_proto_source(${target} "common")
    add_proto_source(${target} "msg")
    add_proto_source(${target} "feder")
    add_proto_source(${target} "rg")
    add_proto_source(${target} "milvus")
    add_proto_service(${target} "milvus")
    target_include_directories(${target} PRIVATE ${milvus_proto_BINARY_DIR})
endfunction(add_milvus_protos target)

