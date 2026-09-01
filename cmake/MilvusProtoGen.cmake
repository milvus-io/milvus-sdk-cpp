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

set(PROTO_COMMIT fd141a092113dd7a85ab6dc6c1b037b17a3bf96a)

include(FetchContent)

# download proto
FetchContent_Declare(milvus_proto
    GIT_REPOSITORY https://github.com/milvus-io/milvus-proto.git
    GIT_TAG        ${PROTO_COMMIT}
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

function(_milvus_get_conan_protoc out_var)
    set(_protobuf_include_dirs)

    # The imported target is the source of truth for the protobuf library that this build links.
    # Check its package root before loose variables, which can retain paths from an earlier
    # module-mode find_package(Protobuf) call.
    if (TARGET protobuf::libprotobuf)
        get_target_property(_target_includes protobuf::libprotobuf INTERFACE_INCLUDE_DIRECTORIES)
        if (_target_includes)
            list(APPEND _protobuf_include_dirs ${_target_includes})
        endif()
    endif()

    # Fallback only for package/module configurations that do not expose target include directories.
    if (NOT _protobuf_include_dirs)
        list(APPEND _protobuf_include_dirs ${protobuf_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIRS})
    endif()
    list(REMOVE_DUPLICATES _protobuf_include_dirs)

    foreach(_inc IN LISTS _protobuf_include_dirs)
        if (NOT _inc)
            continue()
        endif()
        # Conan CMakeDeps may expose config-specific include directories as generator expressions.
        if (_inc MATCHES "^\\$<\\$<CONFIG:[^>]+>:(.*)>$")
            set(_inc "${CMAKE_MATCH_1}")
        endif()
        if (NOT IS_DIRECTORY "${_inc}")
            continue()
        endif()

        get_filename_component(_protobuf_root "${_inc}" DIRECTORY)
        set(_exact_protoc "${_protobuf_root}/bin/protoc${CMAKE_EXECUTABLE_SUFFIX}")
        if (EXISTS "${_exact_protoc}")
            set(${out_var} "${_exact_protoc}" PARENT_SCOPE)
            return()
        endif()

        file(GLOB _versioned_protoc
            LIST_DIRECTORIES FALSE
            "${_protobuf_root}/bin/protoc-*${CMAKE_EXECUTABLE_SUFFIX}")
        if (_versioned_protoc)
            list(SORT _versioned_protoc)
            list(GET _versioned_protoc 0 _protoc)
            set(${out_var} "${_protoc}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
endfunction()

if (BUILD_FROM_CONAN AND NOT CMAKE_CROSSCOMPILING)
    _milvus_get_conan_protoc(_MILVUS_CONAN_PROTOC)
endif()

if (_MILVUS_CONAN_PROTOC)
    set(Protobuf_PROTOC_EXECUTABLE "${_MILVUS_CONAN_PROTOC}" CACHE FILEPATH "The protoc compiler" FORCE)
    set(PROTOC_PROGRAM "${_MILVUS_CONAN_PROTOC}" CACHE FILEPATH "The protoc compiler" FORCE)
    set_property(TARGET protobuf::protoc PROPERTY IMPORTED_LOCATION "${_MILVUS_CONAN_PROTOC}")
    set(_MILVUS_PROTOC_DISPLAY "${_MILVUS_CONAN_PROTOC}")
else()
    get_target_property(_MILVUS_PROTOC_DISPLAY protobuf::protoc IMPORTED_LOCATION)
    if (NOT _MILVUS_PROTOC_DISPLAY)
        set(_MILVUS_PROTOC_DISPLAY "protobuf::protoc target")
    endif()
    if (BUILD_FROM_CONAN AND NOT CMAKE_CROSSCOMPILING)
        message(WARNING
            "Could not locate protoc beside the linked Conan Protobuf package. "
            "Falling back to ${_MILVUS_PROTOC_DISPLAY}, which may be a system-installed compiler.")
    endif()
    set(Protobuf_PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
endif()
set(GRPC_CPP_PLUGIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
message(STATUS "using protoc: ${_MILVUS_PROTOC_DISPLAY}")
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
    # Generated protobuf sources are huge standalone TUs; batching them with hand-written
    # code would amplify memory pressure and can slow the build. The property is a no-op
    # when unity builds are disabled.
    set_source_files_properties(${milvus_proto_BINARY_DIR}/${name}.pb.cc
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION TRUE)
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
    set_source_files_properties(${milvus_proto_BINARY_DIR}/${name}.grpc.pb.cc
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION TRUE)
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
