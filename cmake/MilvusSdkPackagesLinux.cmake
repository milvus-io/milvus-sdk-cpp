# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0(the
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

include(GNUInstallDirs)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set (SUFFIX "64")
    MESSAGE(STATUS "Build System = ${CMAKE_SYSTEM_NAME} - Bitness : 64-bit - Compiler : ${CMAKE_CXX_COMPILER_ID}")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set (SUFFIX "32")
    MESSAGE(STATUS "Build System = ${CMAKE_SYSTEM_NAME} - Bitness : 32-bit - Compiler : ${CMAKE_CXX_COMPILER_ID}")
else()
    MESSAGE(FATAL_ERROR "Could not detect system bitness")
endif()

## debian like
if(EXISTS "/etc/debian_version")
    set(PLATFORM "Debian")
    if(EXISTS "/etc/lsb-release")
        file(READ "/etc/lsb-release" LSB_RELEASE_ID)
        string(REGEX MATCH "DISTRIB_ID=([a-zA-Z0-9 /\\.]+)" _ ${LSB_RELEASE_ID})
        set(FULL_FLAVOUR ${CMAKE_MATCH_1})

        if(FULL_FLAVOUR MATCHES "^.*Ubuntu.*$")
            set(PLATFORM "Ubuntu")
            file(READ "/etc/lsb-release" UBUNTU_RELEASE)
            string(REGEX MATCH "DISTRIB_RELEASE=([0-9 /\\.]+)" _ ${UBUNTU_RELEASE})
            set(PLATFORM_VERSION ${CMAKE_MATCH_1})
        else()
            file(READ "/etc/debian_version" DEBIAN_RELEASE)
            string(REGEX MATCH "([0-9]+\\.[0-9]+)" _ ${DEBIAN_RELEASE})
            set(PLATFORM_VERSION ${CMAKE_MATCH_1})
        endif()
    elseif(EXISTS "/etc/debian_version")
        file(READ "/etc/debian_version" DEBIAN_RELEASE)
        string(REGEX MATCH "([0-9]+\\.[0-9]+)" _ ${DEBIAN_RELEASE})
        set(PLATFORM_VERSION ${CMAKE_MATCH_1})
    endif()
## centos like
elseif(EXISTS "/etc/centos-release")
    set(PLATFORM "CentOS")
    file(READ "/etc/centos-release" CENTOS_RELEASE)
    string(REGEX MATCH "release([0-9 /\\.]+)" _ ${CENTOS_RELEASE})
    set(PLATFORM_VERSION ${CMAKE_MATCH_1})
## others
elseif(EXISTS "/etc/os-release")
    file(READ "/etc/os-release" OS_RELEASE_NAME)
    string(REGEX MATCH "NAME=\"([a-zA-Z0-9 /\\.]+)\"" _ ${OS_RELEASE_NAME})
    set(FULL_PLATFORM ${CMAKE_MATCH_1})
    if(FULL_PLATFORM MATCHES "^.*openSUSE.*$")
        set(PLATFORM "openSUSE")
    elseif(FULL_PLATFORM MATCHES "^.*Ubuntu.*$")
        set(PLATFORM "Ubuntu")
    elseif(FULL_PLATFORM MATCHES "^.*Debian.*$")
        set(PLATFORM "Debian")
    elseif(FULL_PLATFORM MATCHES "^.*CentOS.*$")
        set(PLATFORM "CentOS")
    endif()
    file(READ "/etc/os-release" OS_RELEASE)
    string(REGEX MATCH "VERSION=\"([a-zA-Z0-9 /\\.]+)\"" _ ${OS_RELEASE})
    set(PLATFORM_VERSION ${CMAKE_MATCH_1})
endif()

string(REGEX REPLACE " " "" PLATFORM_VERSION "${PLATFORM_VERSION}")
set(DISTRO_NAME ${PLATFORM}-${PLATFORM_VERSION})
string(TOLOWER ${DISTRO_NAME} distro_name)
MESSAGE(STATUS "Platform = ${PLATFORM}")
MESSAGE(STATUS "Platform Version = ${PLATFORM_VERSION}")
MESSAGE(STATUS "Distribution name = ${DISTRO_NAME}")

if(PLATFORM STREQUAL "Debian" OR PLATFORM STREQUAL "Ubuntu")
    find_program(DPKG_CMD dpkg)
    if(NOT DPKG_CMD)
        if(SUFFIX STREQUAL "32")
            SET(ARCHITECTURE i386)
        elseif(SUFFIX STREQUAL "64")
            SET(ARCHITECTURE amd64)
        endif()
    else()
        execute_process(COMMAND dpkg --print-architecture OUTPUT_VARIABLE ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
elseif(PLATFORM STREQUAL "CentOS" OR PLATFORM STREQUAL "openSUSE")
    execute_process(COMMAND arch OUTPUT_VARIABLE ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

MESSAGE(STATUS "Architecture = ${ARCHITECTURE}")

# install targets, rpm with shared only
install(TARGETS milvus_sdk_shared
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT lib)
install(FILES ${milvus_public_headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT dev)

# Package
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Pack)
set(VENDOR "milvus.io")
set(LICENSE "Apache License")
# set(CONTACT "")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY       "Milvus C++ client runtime for Milvus 2.0+")
set(CPACK_PACKAGE_lib_DESCRIPTION_SUMMARY   "Milvus C++ client runtime for Milvus 2.0+")
set(CPACK_PACKAGE_dev_DESCRIPTION_SUMMARY   "Milvus C++ client development package for Milvus 2.0+")
set(CPACK_PACKAGE_DESCRIPTION               "This package contains Milvus C++ client runtime for Milvus 2.0+")
set(CPACK_PACKAGE_lib_DESCRIPTION           "This package contains Milvus C++ client runtime for Milvus 2.0+")
set(CPACK_PACKAGE_dev_DESCRIPTION           "This package contains Milvus C++ client headers for Milvus 2.0+")
set(CPACK_PACKAGE_NAME                      "libmilvus")
set(CPACK_PACKAGE_VERSION                   "${MILVUS_SDK_VERSION}")
set(CPACK_PACKAGE_RELEASE                   "${MILVUS_SDK_RELEASE}")
set(CPACK_PACKAGE_VENDOR                    ${VENDOR})
set(CPACK_PACKAGE_LICENSE                   ${LICENSE})
set(CPACK_RESOURCE_FILE_LICENSE             "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_CONTACT                   ${CONTACT})
set(CPACK_PACKAGE_CHECKSUM                  SHA256)
set(CPACK_PACKAGE_RELOCATABLE               "OFF")
set(CPACK_PACKAGE_DIRECTORY                 ${CMAKE_CURRENT_BINARY_DIR}/Pack)
set(CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX      "/usr/src/debug/libmilvus")

if (PLATFORM STREQUAL "CentOS" OR PLATFORM STREQUAL "openSUSE")
    set(CPACK_GENERATOR                     "RPM")
    set(CPACK_COMPONENTS_GROUPING           ONE_PER_GROUP)
    set(CPACK_RPM_COMPONENT_INSTALL         ON)
    set(CPACK_RPM_PACKAGE_LICENSE           ${CPACK_PACKAGE_LICENSE})
    set(CPACK_RPM_PACKAGE_GROUP             "Development/Libraries")
    set(CPACK_RPM_PACKAGE_VENDOR            ${CPACK_PACKAGE_VENDOR})
    set(CPACK_RPM_PACKAGE_AUTOREQ           "yes")
    set(CPACK_RPM_PACKAGE_RELOCATABLE       "OFF")
    set(CPACK_RPM_PACKAGE_ARCHITECTURE      "${ARCHITECTURE}")

    set(CPACK_RPM_lib_PACKAGE_NAME          "${CPACK_PACKAGE_NAME}")
    set(CPACK_RPM_lib_FILE_NAME             "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}-${distro_name}-${ARCHITECTURE}.rpm")
    set(CPACK_RPM_lib_PACKAGE_SUMMARY       ${CPACK_PACKAGE_lib_DESCRIPTION_SUMMARY})
    set(CPACK_RPM_lib_PACKAGE_DESCRIPTION   ${CPACK_PACKAGE_lib_DESCRIPTION})
    set(CPACK_RPM_lib_DEBUGINFO_PACKAGE     ON)

    set(CPACK_RPM_dev_PACKAGE_NAME          "${CPACK_PACKAGE_NAME}-devel")
    set(CPACK_RPM_dev_FILE_NAME             "${CPACK_PACKAGE_NAME}-devel-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}-${distro_name}-${ARCHITECTURE}.rpm")
    set(CPACK_RPM_dev_PACKAGE_SUMMARY       ${CPACK_PACKAGE_dev_DESCRIPTION_SUMMARY})
    set(CPACK_RPM_dev_PACKAGE_DESCRIPTION   ${CPACK_PACKAGE_dev_DESCRIPTION})
endif()

include(CPack)
