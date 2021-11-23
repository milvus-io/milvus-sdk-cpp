#!/usr/bin/env bash

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

set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
ROOT_DIR="$( cd -P "$( dirname "$SOURCE" )/.." && pwd )"
echo "ROOT_DIR=" ${ROOT_DIR}

GRPC_BIN_PATH="${ROOT_DIR}/cmake_build/grpc_ep-prefix/src/grpc_ep/bins"
PROTOC_PATH="${GRPC_BIN_PATH}/opt/protobuf/protoc"
GRPC_PULGIN_PATH="${GRPC_BIN_PATH}}/opt/grpc_cpp_plugin"

PROTO_PATH=${ROOT_DIR}/src/proto/
OUTPUT_PATH=${ROOT_DIR}/src/proto-gen/

echo "GRPC_BIN_PATH=" ${GRPC_BIN_PATH}
echo "PROTO_PATH=" ${PROTOC_PATH}
echo "OUTPUT_PATH=" ${OUTPUT_PATH}

${PROTOC_PATH} -I . --grpc_out=${OUTPUT_PATH} --plugin=protoc-gen-grpc="${GRPC_PULGIN_PATH}" ${PROTO_PATH}/common.proto

${PROTOC_PATH} -I . --cpp_out=${OUTPUT_PATH} ${PROTO_PATH}/common.proto

${PROTOC_PATH} -I . --grpc_out=${OUTPUT_PATH} --plugin=protoc-gen-grpc="${GRPC_PULGIN_PATH}" ${PROTO_PATH}/milvus.proto

${PROTOC_PATH} -I . --cpp_out=${OUTPUT_PATH} ${PROTO_PATH}/milvus.proto

${PROTOC_PATH} -I . --grpc_out=${OUTPUT_PATH} --plugin=protoc-gen-grpc="${GRPC_PULGIN_PATH}" ${PROTO_PATH}/schema.proto

${PROTOC_PATH} -I . --cpp_out=${OUTPUT_PATH} ${PROTO_PATH}/schema.proto