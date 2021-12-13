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

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
ROOT_DIR="$( cd -P "$( dirname "$SOURCE" )/.." && pwd )"

echo "ROOT_DIR = ${ROOT_DIR}"

DIR_LCOV_OUTPUT="${ROOT_DIR}/${COVERAGE_OUTPUT_DIR}"
DIR_GCNO="${ROOT_DIR}/${BUILD_OUTPUT_DIR}/"
FILE_INFO_BASE="${DIR_LCOV_OUTPUT}/lcov_base.info"
FILE_INFO_UT="${DIR_LCOV_OUTPUT}/lcov_ut.info"
FILE_INFO_COMBINE="${DIR_LCOV_OUTPUT}/lcov_combine.info"
FILE_INFO_OUTPUT="${DIR_LCOV_OUTPUT}/lcov_output.info"

# delete old code coverage output files
rm -rf ${DIR_LCOV_OUTPUT}
mkdir ${COVERAGE_OUTPUT_DIR}

# generate baseline
${LCOV_CMD} -c -i -d ${DIR_GCNO} -o ${FILE_INFO_BASE}
if [ $? -ne 0 ]; then
    echo "Failed to generate coverage baseline"
    exit -1
fi

# generate ut file
${LCOV_CMD} -c -d ${DIR_GCNO} -o ${FILE_INFO_UT}

# merge baseline and ut file
${LCOV_CMD} -a ${FILE_INFO_BASE} -a ${FILE_INFO_UT} -o ${FILE_INFO_COMBINE}

# remove unnecessary info
${LCOV_CMD} -r "${FILE_INFO_COMBINE}" -o "${FILE_INFO_OUTPUT}" \
    "/usr/*" \
    "*/install/*" \
    "*/third_party/*" \
    "*/src/proto-gen/*" \
    "*/test/*" \
    "*/_deps/*" \
    "*/examples/*"

# generate html report
${LCOV_GEN_CMD} ${FILE_INFO_OUTPUT} --output-directory ${DIR_LCOV_OUTPUT}/
echo "Generate cpp code coverage report to ${DIR_LCOV_OUTPUT}"
