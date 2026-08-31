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
# Set COVERAGE_HTML=OFF to skip the genhtml step (lcov_output.info is still
# generated). CI uses this: codecov only consumes the .info tracefile and the
# HTML tree dies with the ephemeral runner anyway. Local builds keep HTML.
COVERAGE_HTML=${COVERAGE_HTML:-ON}
# lcov 1.x runs gcov sequentially per object file, which makes capture the
# slowest part of the report pipeline. Capture chunks are balanced at data-file
# granularity (a single large object cannot pin the critical path) and merged
# afterwards; the filtered output is byte-identical to the sequential run.
COVERAGE_JOBS=${COVERAGE_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}

case "${COVERAGE_HTML}" in
    ON|OFF) ;;
    *)
        echo "ERROR! COVERAGE_HTML must be ON or OFF"
        exit 1
        ;;
esac

if ! [[ "${COVERAGE_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR! COVERAGE_JOBS must be a positive integer"
    exit 1
fi

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

# Capture lcov data for the given pattern (*.gcda for test data, *.gcno with
# -i for the baseline), either sequentially or across COVERAGE_JOBS workers.
capture_coverage() {
    local extra_flag=$1
    local pattern=$2
    local output=$3
    local -a files=()
    while IFS= read -r f; do
        files+=("$f")
    done < <(find "${DIR_GCNO}" -name "${pattern}" | sort)

    if [[ "${COVERAGE_JOBS}" -le 1 || ${#files[@]} -le "${COVERAGE_JOBS}" ]]; then
        ${LCOV_CMD} --gcov-tool "${GCOV_TOOL}" -c ${extra_flag} -d "${DIR_GCNO}" -o "${output}" \
            --exclude "*/_deps/*" --exclude "*.pb.cc" --exclude "*.grpc.pb.cc"
        return $?
    fi

    local chunk_root
    chunk_root=$(mktemp -d)
    local -a loads pids merge_args
    local j f size min_j
    for j in $(seq 0 $((COVERAGE_JOBS - 1))); do
        loads[$j]=0
        : > "${chunk_root}/chunk_${j}.txt"
    done

    # greedy assignment by file size keeps every worker busy
    for f in "${files[@]}"; do
        size=$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")
        min_j=0
        for j in $(seq 1 $((COVERAGE_JOBS - 1))); do
            if [[ ${loads[$j]} -lt ${loads[$min_j]} ]]; then
                min_j=$j
            fi
        done
        loads[$min_j]=$(( loads[$min_j] + size ))
        echo "$f" >> "${chunk_root}/chunk_${min_j}.txt"
    done

    for j in $(seq 0 $((COVERAGE_JOBS - 1))); do
        local -a dargs=()
        while IFS= read -r d; do
            dargs+=(-d "$d")
        done < "${chunk_root}/chunk_${j}.txt"
        ${LCOV_CMD} --gcov-tool "${GCOV_TOOL}" -c ${extra_flag} "${dargs[@]}" \
            -o "${chunk_root}/part_${j}.info" \
            --exclude "*/_deps/*" --exclude "*.pb.cc" --exclude "*.grpc.pb.cc" \
            > "${chunk_root}/part_${j}.log" 2>&1 &
        pids[$j]=$!
    done

    local failed=0
    for j in $(seq 0 $((COVERAGE_JOBS - 1))); do
        if ! wait "${pids[$j]}"; then
            echo "lcov worker ${j} failed:"
            cat "${chunk_root}/part_${j}.log"
            failed=1
        fi
    done
    if [[ ${failed} -ne 0 ]]; then
        rm -rf "${chunk_root}"
        return 1
    fi

    # A chunk whose files are all excluded (e.g. generated protobuf sources)
    # yields a part file with no records, and `lcov -a` aborts on it; skip
    # empty parts so the merge stays robust for any chunk layout.
    local j_skip=0
    for j in $(seq 0 $((COVERAGE_JOBS - 1))); do
        if grep -q '^SF:' "${chunk_root}/part_${j}.info" 2>/dev/null; then
            merge_args+=(-a "${chunk_root}/part_${j}.info")
        else
            echo "Skipping lcov chunk ${j}: no records after exclusions"
            j_skip=$((j_skip + 1))
        fi
    done
    if [[ ${j_skip} -eq ${COVERAGE_JOBS} ]]; then
        rm -rf "${chunk_root}"
        return 1
    fi

    ${LCOV_CMD} --gcov-tool "${GCOV_TOOL}" "${merge_args[@]}" -o "${output}"
    local status=$?
    rm -rf "${chunk_root}"
    return ${status}
}

# delete old code coverage output files
rm -rf ${DIR_LCOV_OUTPUT}
mkdir ${COVERAGE_OUTPUT_DIR}

# generate baseline (exclude proto-generated files)
if ! capture_coverage -i '*.gcno' "${FILE_INFO_BASE}"; then
    echo "Failed to generate coverage baseline"
    exit -1
fi

# generate ut file (exclude proto-generated files)
if ! capture_coverage '' '*.gcda' "${FILE_INFO_UT}"; then
    echo "Failed to generate coverage data"
    exit -1
fi

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
    "*/tutorial/*" \
    "*/examples/*" \
    "*/.conan2/*"

# generate html report
if [[ "${COVERAGE_HTML}" == "ON" ]]; then
    ${LCOV_GEN_CMD} ${FILE_INFO_OUTPUT} --output-directory ${DIR_LCOV_OUTPUT}/
    echo "Generate cpp code coverage report to ${DIR_LCOV_OUTPUT}"
else
    echo "Skip html report generation (COVERAGE_HTML=OFF); ${FILE_INFO_OUTPUT} is unchanged"
fi
