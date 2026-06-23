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

BUILD_OUTPUT_DIR="cmake_build"
COVERAGE_OUTPUT_DIR="code_coverage"

LCOV_CMD="lcov"
LCOV_GEN_CMD="genhtml"

resolve_gcov_tool() {
    if [ -n "${GCOV_TOOL}" ] ; then
        echo "${GCOV_TOOL}"
        return 0
    fi

    local compiler_file
    compiler_file=$(find "${ROOT_DIR}/${BUILD_OUTPUT_DIR}/CMakeFiles" -path '*/CMakeCXXCompiler.cmake' | head -n 1)
    if [ -n "${compiler_file}" ] ; then
        local compiler_version gcov_candidate
        compiler_version=$(grep 'set(CMAKE_CXX_COMPILER_VERSION "' "${compiler_file}" | sed -E 's/.*"([0-9]+)\..*/\1/' | head -n 1)
        if [ -n "${compiler_version}" ] ; then
            gcov_candidate="gcov-${compiler_version}"
            if command -v "${gcov_candidate}" >/dev/null 2>&1 ; then
                echo "${gcov_candidate}"
                return 0
            fi
        fi
    fi

    echo "gcov"
}

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
ROOT_DIR="$( cd -P "$( dirname "$SOURCE" )/.." && pwd )"
GCOV_TOOL="$(resolve_gcov_tool)"

echo "ROOT_DIR = ${ROOT_DIR}"
echo "GCOV_TOOL = ${GCOV_TOOL}"

DIR_LCOV_OUTPUT="${ROOT_DIR}/${COVERAGE_OUTPUT_DIR}"
DIR_GCNO="${ROOT_DIR}/${BUILD_OUTPUT_DIR}/src/"
FILE_INFO_BASE="${DIR_LCOV_OUTPUT}/lcov_base.info"
FILE_INFO_UT="${DIR_LCOV_OUTPUT}/lcov_ut.info"
FILE_INFO_COMBINE="${DIR_LCOV_OUTPUT}/lcov_combine.info"
FILE_INFO_OUTPUT="${DIR_LCOV_OUTPUT}/lcov_output.info"

# delete old code coverage output files
rm -rf ${DIR_LCOV_OUTPUT}
mkdir ${COVERAGE_OUTPUT_DIR}

# generate baseline (exclude proto-generated files)
${LCOV_CMD} --gcov-tool "${GCOV_TOOL}" -c -i -d ${DIR_GCNO} -o ${FILE_INFO_BASE} \
    --exclude "*/_deps/*" --exclude "*.pb.cc" --exclude "*.grpc.pb.cc"
if [ $? -ne 0 ]; then
    echo "Failed to generate coverage baseline"
    exit -1
fi

# generate ut file (exclude proto-generated files)
${LCOV_CMD} --gcov-tool "${GCOV_TOOL}" -c -d ${DIR_GCNO} -o ${FILE_INFO_UT} \
    --exclude "*/_deps/*" --exclude "*.pb.cc" --exclude "*.grpc.pb.cc"

# merge baseline and ut file
${LCOV_CMD} --gcov-tool "${GCOV_TOOL}" -a ${FILE_INFO_BASE} -a ${FILE_INFO_UT} -o ${FILE_INFO_COMBINE}

# remove unnecessary info
${LCOV_CMD} --gcov-tool "${GCOV_TOOL}" -r "${FILE_INFO_COMBINE}" -o "${FILE_INFO_OUTPUT}" \
    "/usr/*" \
    "*/install/*" \
    "*/src/include/nlohmann/*" \
    "*/thirdparty/*" \
    "*/test/*" \
    "*/_deps/*" \
    "*/examples/*" \
    "*/.conan2/*"

# generate html report
${LCOV_GEN_CMD} ${FILE_INFO_OUTPUT} --output-directory ${DIR_LCOV_OUTPUT}/
echo "Generate cpp code coverage report to ${DIR_LCOV_OUTPUT}"
