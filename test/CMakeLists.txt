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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/impl)
include_directories(${milvus_proto_BINARY_DIR})

# GTest is provided in different ways depending on the build mode:
# - BUILD_FROM_CONAN=ON: Conan CMakeDeps provides GTestConfig.cmake and imported targets.
# - non-Conan builds (scripts/build.sh -z): we rely on ThirdPartyPackages.cmake to fetch/build googletest.
if ("${BUILD_FROM_CONAN}" STREQUAL "ON")
    find_package(GTest CONFIG REQUIRED)
endif()

set(GTEST_LIBRARIES GTest::gtest)
set(GMOCK_LIBRARIES GTest::gmock)
set(GTEST_MAIN_LIBRARIES GTest::gtest_main)

set(UT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ut")
file(GLOB_RECURSE ut_files
    "${UT_DIR}/*.cpp"
    "${UT_DIR}/*.cxx"
    "${UT_DIR}/*.cc"
)
add_executable(testing-ut ${ut_files})
target_compile_options(testing-ut PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
target_link_libraries(testing-ut PRIVATE milvus_sdk ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

set(IT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/it")
file(GLOB_RECURSE it_files
    "${IT_DIR}/*.cpp"
    "${IT_DIR}/*.cxx"
    "${IT_DIR}/*.cc"
)
add_executable(testing-it ${it_files})
target_compile_options(testing-it PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
target_link_libraries(testing-it PRIVATE milvus_sdk ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

# st only available under linux/macos
if (CMAKE_SYSTEM_NAME MATCHES "(Linux|Darwin)")
set(ST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/st")
file(GLOB_RECURSE st_files
    "${ST_DIR}/*.cpp"
    "${ST_DIR}/*.cxx"
    "${ST_DIR}/*.cc"
)
add_executable(testing-st ${st_files})
target_link_libraries(testing-st PRIVATE milvus_sdk ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES})
endif()
