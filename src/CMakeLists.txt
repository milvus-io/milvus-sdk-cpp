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


include_directories(include)
include_directories(proto-gen)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/impl impl_files)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/impl/types impl_types_files)
include(proto-gen/proto-gen.cmake) # milvus_sdk_proto_files

# static library
if(${BUILD_STATIC})
add_library(milvus_sdk_static STATIC
        ${impl_files}
        ${impl_types_files}
        ${milvus_sdk_proto_files}
        )
set_target_properties(milvus_sdk_static PROPERTIES OUTPUT_NAME milvus_sdk)
target_link_libraries(milvus_sdk_static grpc++)
add_dependencies(milvus_sdk_static protoc grpc_cpp_plugin)
set(milvus_sdk_internal milvus_sdk_static)
endif()

# shared library
if(${BUILD_SHARED})
add_library(milvus_sdk_shared SHARED
        ${impl_files}
        ${impl_types_files}
        ${milvus_sdk_proto_files}
        )
set_target_properties(milvus_sdk_shared PROPERTIES OUTPUT_NAME milvus_sdk)
target_link_libraries(milvus_sdk_shared grpc++)
add_dependencies(milvus_sdk_shared protoc grpc_cpp_plugin)
set(milvus_sdk_internal milvus_sdk_shared)
endif()

# using milvus_sdk shared first, then static
add_library(milvus_sdk INTERFACE)
target_link_libraries(milvus_sdk INTERFACE ${milvus_sdk_internal})

# package
include(MilvusSdkPackages)