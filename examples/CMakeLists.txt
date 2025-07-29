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

cmake_minimum_required(VERSION 3.12)
project(milvus_sdk_example LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

message(STATUS "  CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
include_directories(${CMAKE_SOURCE_DIR}/src/include)
include_directories(${CMAKE_SOURCE_DIR}/examples)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/array array_files)
add_executable(sdk_array ${array_files})
target_link_libraries(sdk_array milvus_sdk)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/db db_files)
add_executable(sdk_db ${db_files})
target_link_libraries(sdk_db milvus_sdk)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/dml dml_files)
add_executable(sdk_dml ${dml_files})
target_link_libraries(sdk_dml milvus_sdk)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/json json_files)
add_executable(sdk_json ${json_files})
target_link_libraries(sdk_json milvus_sdk)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/simple simple_files)
add_executable(sdk_simple ${simple_files})
target_link_libraries(sdk_simple milvus_sdk)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/vector_binary binary_files)
add_executable(sdk_binary ${binary_files})
target_link_libraries(sdk_binary milvus_sdk)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/vector_sparse sparse_files)
add_executable(sdk_sparse ${sparse_files})
target_link_libraries(sdk_sparse milvus_sdk)
