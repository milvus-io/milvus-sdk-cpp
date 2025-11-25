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

cmake_minimum_required(VERSION 3.5)
project(milvus_sdk_example LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)

message(STATUS "  CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
include_directories(${CMAKE_SOURCE_DIR}/src/include)

set(example_utils ${CMAKE_CURRENT_SOURCE_DIR}/utils)

function(build_executable src_path ver)
    get_filename_component(filename ${src_path} NAME_WE)
    set(exe_name sdk_${filename}_${ver})
    add_executable(${exe_name} ${src_path})
    set_target_properties(${exe_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/${ver}"
    )
    target_include_directories(${exe_name} PRIVATE ${example_utils})
    target_link_libraries(${exe_name} milvus_sdk)
endfunction()

file(GLOB src_files_v1 ${CMAKE_CURRENT_SOURCE_DIR}/src/v1/*.cpp)
foreach(src_file ${src_files_v1})
    build_executable(${src_file} "v1")
endforeach()

file(GLOB src_files_v2 ${CMAKE_CURRENT_SOURCE_DIR}/src/v2/*.cpp)
foreach(src_file ${src_files_v2})
    build_executable(${src_file} "v2")
endforeach()
