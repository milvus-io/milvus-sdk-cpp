
include_directories(include)
include_directories(proto-gen)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/impl impl_files)
include(proto-gen/proto-gen.cmake) # milvus_sdk_proto_files

set(grpc_lib
        grpcpp_channelz
        grpc++
        grpc
        grpc_protobuf
        grpc_protoc
        )

set(third_party_libs
        zlib
        )

add_library(milvus_sdk SHARED
        ${impl_files}
        ${milvus_sdk_proto_files}
        )

target_link_libraries(milvus_sdk
        ${grpc_lib}
        zlib
        dl
        z
        )

install(TARGETS milvus_sdk DESTINATION lib)
