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
include(proto-gen/proto-gen.cmake) # milvus_sdk_proto_files

# static library
if(${BUILD_STATIC})
add_library(milvus_sdk_static STATIC
        ${impl_files}
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

# public headers
set(milvus_public_headers
    include/milvus/param/BaseParam.h
    include/milvus/types/ConnectParam.h
    include/milvus/types/DataType.h
    include/milvus/types/CollectionStat.h
    include/milvus/types/PartitionStat.h
    include/milvus/types/IndexState.h
    include/milvus/types/PartitionInfo.h
    include/milvus/types/IDArray.h
    include/milvus/types/QueryArguments.h
    include/milvus/types/QueryResults.h
    include/milvus/types/CollectionInfo.h
    include/milvus/types/IndexDesc.h
    include/milvus/types/SegmentInfo.h
    include/milvus/types/CompactionPlan.h
    include/milvus/types/CompactionState.h
    include/milvus/types/Constants.h
    include/milvus/types/FieldSchema.h
    include/milvus/types/ProgressMonitor.h
    include/milvus/types/CollectionSchema.h
    include/milvus/types/MetricType.h
    include/milvus/types/SearchResults.h
    include/milvus/types/CollectionDesc.h
    include/milvus/types/FieldData.h
    include/milvus/types/SearchArguments.h
    include/milvus/Status.h
    include/milvus/MilvusClient.h
)

# package
include(MilvusSdkPackages)