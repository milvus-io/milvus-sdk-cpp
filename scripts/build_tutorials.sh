#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_NAME=${CONAN_REMOTE_NAME:-default-conan-local2}
REMOTE_URL=${CONAN_REMOTE_URL:-https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2}
JOBS=${JOBS:-4}
SDK_VERSION=${TUTORIAL_SDK_VERSION:-0.0}
SDK_USER=${TUTORIAL_SDK_USER:-milvus-tutorial-ci}
SDK_CHANNEL=${TUTORIAL_SDK_CHANNEL:-testing}

if ! command -v conan >/dev/null 2>&1; then
    echo "ERROR: Conan 2 is required to build the standalone tutorials." >&2
    exit 1
fi

if ! conan remote list | grep -q "^${REMOTE_NAME}:"; then
    conan remote add "${REMOTE_NAME}" "${REMOTE_URL}" --allowed-packages "milvus-sdk-cpp/*"
fi

echo "Exporting milvus-sdk-cpp/${SDK_VERSION}@${SDK_USER}/${SDK_CHANNEL} from the current checkout"
conan export "${ROOT_DIR}" \
    --version="${SDK_VERSION}" \
    --user="${SDK_USER}" \
    --channel="${SDK_CHANNEL}"

# Build the SDK package only when the current checkout is not already cached.
# conan create always forces a rebuild; export + install --build=missing reuses
# the cached package whenever the recipe revision is unchanged.
SDK_REF="milvus-sdk-cpp/${SDK_VERSION}@${SDK_USER}/${SDK_CHANNEL}"
SDK_BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "${SDK_BUILD_DIR}"' EXIT
conan install --requires="${SDK_REF}" \
    --build=missing \
    -s build_type=Release \
    -s compiler.cppstd=14 \
    -s:b build_type=Release \
    -s:b compiler.cppstd=17 \
    -c tools.build:jobs="${JOBS}" \
    --output-folder="${SDK_BUILD_DIR}"

export MILVUS_SDK_VERSION="${SDK_VERSION}"
export MILVUS_SDK_USER="${SDK_USER}"
export MILVUS_SDK_CHANNEL="${SDK_CHANNEL}"

for tutorial_dir in "${ROOT_DIR}"/tutorial/[0-9]_*; do
    echo "Building ${tutorial_dir#"${ROOT_DIR}"/}"
    make -C "${tutorial_dir}" JOBS="${JOBS}" build
done
