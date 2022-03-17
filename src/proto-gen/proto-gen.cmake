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

function(add_proto_source name)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/proto-gen/${name}.pb.cc
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/milvus-proto/proto/${name}.proto
                ${protobuf_BINARY_DIR}/protoc${CMAKE_EXECUTABLE_SUFFIX}
        COMMAND ${protobuf_BINARY_DIR}/protoc${CMAKE_EXECUTABLE_SUFFIX}
                --cpp_out ${CMAKE_CURRENT_SOURCE_DIR}/proto-gen -I${CMAKE_CURRENT_SOURCE_DIR}/milvus-proto/proto
                ${CMAKE_CURRENT_SOURCE_DIR}/milvus-proto/proto/${name}.proto
    )
endfunction(add_proto_source name)

function(add_proto_service name)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/proto-gen/${name}.grpc.pb.cc
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/milvus-proto/proto/${name}.proto
                ${protobuf_BINARY_DIR}/protoc${CMAKE_EXECUTABLE_SUFFIX}
                ${grpc_BINARY_DIR}/grpc_cpp_plugin${CMAKE_EXECUTABLE_SUFFIX}
        COMMAND ${protobuf_BINARY_DIR}/protoc${CMAKE_EXECUTABLE_SUFFIX}
                --grpc_out ${CMAKE_CURRENT_SOURCE_DIR}/proto-gen -I${CMAKE_CURRENT_SOURCE_DIR}/milvus-proto/proto
                --plugin=protoc-gen-grpc=${grpc_BINARY_DIR}/grpc_cpp_plugin${CMAKE_EXECUTABLE_SUFFIX}
                ${CMAKE_CURRENT_SOURCE_DIR}/milvus-proto/proto/${name}.proto
    )
endfunction(add_proto_service name)


add_proto_source(common)
add_proto_source(schema)
add_proto_source(milvus)
add_proto_service(milvus)

set(milvus_sdk_proto_files
    ${CMAKE_CURRENT_SOURCE_DIR}/proto-gen/common.pb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/proto-gen/schema.pb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/proto-gen/milvus.pb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/proto-gen/milvus.grpc.pb.cc
)

