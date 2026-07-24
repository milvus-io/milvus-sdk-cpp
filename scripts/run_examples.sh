#!/usr/bin/env bash

# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

export GRPC_VERBOSITY=ERROR

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

version="${1:-v2}"
if [[ $# -gt 1 || ("${version}" != "v1" && "${version}" != "v2") ]]; then
    echo "Usage: $0 [v1|v2]" >&2
    exit 2
fi

readonly EXAMPLE_DIR="${REPO_ROOT}/cmake_build/examples/${version}"
if [[ ! -d "${EXAMPLE_DIR}" ]]; then
    echo "Example directory does not exist: ${EXAMPLE_DIR}" >&2
    echo "Build the examples first, for example with: make test" >&2
    exit 1
fi

mapfile -d '' all_examples < <(find "${EXAMPLE_DIR}" -maxdepth 1 -type f -name "sdk_*_${version}" \
    -perm -u+x -print0 | sort -z)
if [[ ${#all_examples[@]} -eq 0 ]]; then
    echo "No executable ${version} examples found in: ${EXAMPLE_DIR}" >&2
    echo "Build the examples first, for example with: make test" >&2
    exit 1
fi

examples=()
for example in "${all_examples[@]}"; do
    example_name="$(basename "${example}")"
    case "${example_name}" in
        sdk_cdc_v2)
            echo "Skipping ${example_name} because it requires two Milvus servers."
            ;;
        sdk_external_table_v2)
            echo "Skipping ${example_name} because it requires pre-populated external object storage."
            ;;
        sdk_optimize_v2)
            echo "Skipping ${example_name} because it is excluded from the batch run."
            ;;
        *)
            examples+=("${example}")
            ;;
    esac
done

if [[ ${#examples[@]} -eq 0 ]]; then
    echo "No runnable ${version} examples remain after applying the skip list." >&2
    exit 1
fi

cd "${REPO_ROOT}"
echo "Running ${#examples[@]} ${version} examples from ${EXAMPLE_DIR}"

for example in "${examples[@]}"; do
    echo
    echo "Running $(basename "${example}")"
    if "${example}"; then
        echo "Passed: $(basename "${example}")"
    else
        status=$?
        echo "Failed: $(basename "${example}") (exit code ${status})" >&2
        exit "${status}"
    fi
done

echo
echo "All runnable ${version} examples passed."
