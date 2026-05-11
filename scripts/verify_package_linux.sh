#!/bin/sh

set -eu

package_dir=${1:-cmake_build/Pack}

package=""
for candidate in "${package_dir}"/libmilvus-*.tar.gz "${package_dir}"/libmilvus-*.deb "${package_dir}"/libmilvus-*.rpm; do
    if [ -f "${candidate}" ]; then
        package="${candidate}"
        break
    fi
done
if [ -z "${package}" ]; then
    echo "No package found under ${package_dir}"
    find "${package_dir}" -type f -print
    exit 1
fi
package="$(cd "$(dirname "${package}")" && pwd)/$(basename "${package}")"

smoke_dir="${RUNNER_TEMP:-/tmp}/milvus-sdk-package-smoke"
rm -rf "${smoke_dir}"
mkdir -p "${smoke_dir}/package"
case "${package}" in
    *.tar.gz)
        tar -xzf "${package}" -C "${smoke_dir}/package" --strip-components=1
        ;;
    *.deb)
        dpkg-deb -x "${package}" "${smoke_dir}/package"
        ;;
    *.rpm)
        (cd "${smoke_dir}/package" && rpm2cpio "${package}" | cpio -idm)
        ;;
    *)
        echo "Unsupported package format: ${package}"
        exit 1
        ;;
esac

include_dir="$(find "${smoke_dir}/package" -type d -path '*/include/milvus' -print -quit)"
lib_dir="$(find "${smoke_dir}/package" -type f \( -name 'libmilvus_sdk.so' -o -name 'libmilvus_sdk.dylib' -o -name 'libmilvus_sdk.a' \) -exec dirname {} \; -quit)"
if [ -z "${include_dir}" ] || [ -z "${lib_dir}" ]; then
    echo "Package does not contain expected include/lib layout"
    find "${smoke_dir}/package" -type f -print | head -200
    exit 1
fi
include_dir="${include_dir%/milvus}"

cat > "${smoke_dir}/smoke.cpp" <<'CPP'
#include <iostream>
#include <string>

#include "milvus/MilvusClientV2.h"

int
main() {
    auto client = milvus::MilvusClientV2::Create();
    std::string version;
    auto status = client->GetSDKVersion(version);
    if (!status.IsOk()) {
        std::cerr << status.Message() << std::endl;
        return 1;
    }
    std::cout << version << std::endl;
    return version.empty() ? 1 : 0;
}
CPP

c++ -std=c++14 "${smoke_dir}/smoke.cpp" \
    -I"${include_dir}" \
    -L"${lib_dir}" \
    -lmilvus_sdk \
    -Wl,-rpath,"${lib_dir}" \
    -o "${smoke_dir}/smoke"
export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="${lib_dir}:${DYLD_LIBRARY_PATH:-}"
"${smoke_dir}/smoke"
