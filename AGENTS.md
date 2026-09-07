# AGENTS.md

Repository guidance for AI agents and contributors working in `milvus-sdk-cpp`.

## Repository layout

The repo builds one shared/static library (`libmilvus_sdk`) plus test, example, and tutorial targets:

| Area | Description | Location |
|---|---|---|
| Public headers | V2 aggregate header plus V1 legacy entry point | `src/include/milvus/` |
| Implementation | V1/V2 impl, connection, conversion utils, caches | `src/impl/` |
| Unit tests | no server required | `test/ut/` |
| Mocked tests | in-process mocked gRPC server | `test/it/` |
| System tests | real Milvus via Docker | `test/st/` |
| Examples | V1 and V2 example programs | `examples/src/{v1,v2}/` |
| Tutorials | standalone apps pinning the published SDK version | `tutorial/` |

- Consumers include only the aggregate header `milvus/MilvusClientV2.h`. When exposing a new public V2 DTO,
  add its header there.
- Request DTOs live in `src/include/milvus/request/<domain>/`, response DTOs in
  `src/include/milvus/response/<domain>/`, shared value types in `src/include/milvus/types/`. Each has a
  matching implementation under `src/impl/request|response|types/`.
- proto→SDK / SDK→proto conversions live in `src/impl/utils/` (`TypeUtils.cpp`, `DmlUtils.cpp`, `DqlUtils.cpp`,
  `MiscUtils.cpp`). Shared machinery: `RpcUtils.cpp` (retry), `ConnectionHandler.cpp` (per-API
  validate/pre/rpc/post pipeline), and `cache/` (`SchemaCache`, `CollectionTsCache`).

## SDK generations

Two API generations coexist:

- **V1 (legacy, maintenance mode)** — `MilvusClient` (`milvus/MilvusClient.h`), implemented in
  `src/impl/MilvusClientImpl.cpp`. Kept for backward compatibility: bug fixes and critical maintenance only,
  **no new features**.
- **V2 (active, long-term)** — `MilvusClientV2` (`milvus/MilvusClientV2.h`), implemented in
  `src/impl/MilvusClientV2Impl.cpp` (+ `MilvusClientV2SessionImpl.cpp` for sessions). Request/response DTO
  pattern: each operation is a `*Request` / `*Response` pair. All new features, API additions, and pymilvus
  parity work target V2.

## Conventions

- **Public SDK APIs do not throw.** Return `Status`; catch and convert exception-prone conversions (for
  example `nlohmann::json::parse`) into a non-OK `Status` instead of letting exceptions escape.
- Request DTOs expose a fluent API: `WithXxx(...)` returning the request reference plus plain
  `SetXxx(...)` / `Xxx()` accessors, mirroring the existing request classes.
- Keep the public declarations, implementation declarations, DTO headers/implementations, proto conversions,
  examples, and tests synchronized for every change.
- Cache-aware RPC paths use one consistent endpoint, resolved database, collection/alias name, and schema load
  scope. Audit successful DDL mutation callbacks for schema/timestamp invalidation or transfer
  (`SchemaCache`, `CollectionTsCache`).
- Do not edit generated protobuf files under the build tree; change the SDK conversion code or proto inputs
  (`_deps/milvus_proto-*`).
- Do not add comments unless they carry real context; keep new code free of stray comments.

## Building

Dependencies come from [Conan 2](https://conan.io/) (gRPC, protobuf, abseil, etc.); `scripts/build.sh`
handles Conan integration and CMake configuration. ccache is enabled automatically via
`RULE_LAUNCH_COMPILE`.

```bash
bash scripts/install_deps.sh     # one-time dev dependency setup
make                             # debug build
make all-release                 # release build
make test                        # debug build + unit/mocked tests
make test-release                # release build + unit/mocked tests
make package                     # release build + DEB/RPM under cmake_build/Pack
make coverage                    # lcov coverage (starts real-server system tests)
```

Useful controls (passed through to CMake):

```bash
JOBS=4 CPPSTD=17 BUILD_SHARED_LIBS=OFF make test
UNITY=ON LINE=ON JOBS=4 make test       # faster cold builds; see DEVELOPMENT.md
CMAKE_BUILD_EXAMPLES=OFF JOBS=4 make test
MILVUS_SDK_VERSION=vX.Y.Z JOBS=4 make all-release
```

`scripts/build.sh` flags: `-t Debug|Release`, `-u` (unit tests), `-l` (lint), `-r` (clean first), `-z`
(no Conan), `-f` (skip in-place clang-format). For an existing configured tree, prefer direct targets:
`cmake --build cmake_build -j4 --target milvus_sdk` / `--target testing-ut` / `--target testing-it`.
Limit builds to four jobs.

## Lint / format

- `make lint` (or `make lint-release`) runs cpplint, clang-format checks, and clang-tidy via
  `scripts/build.sh -l`.
- The lint build forces `UNITY=OFF` and `PCH=OFF` so clang-tidy sees per-file compile commands and can read
  the sources (it cannot consume a GCC precompiled header).
- `make clang-format` inside `cmake_build` reformats all C++ sources in place. Avoid broad reformatting that
  hides unrelated changes.

## Testing

Three layers; the fast layers require no Milvus server. Binaries land under `cmake_build/test/`.

| Scope | Binary | Depends on | Covers |
|---|---|---|---|
| Unit | `testing-ut` | none | DTOs, conversion helpers, caches, retry internals |
| Mocked | `testing-it` | in-process mocked gRPC server | request wiring, error mapping, caches, retry, callbacks |
| System | `testing-st` | real Milvus container via Docker | end-to-end server behavior |

Run focused cases with:

```bash
GRPC_VERBOSITY=ERROR ./cmake_build/test/testing-ut --gtest_filter='*CollectionDesc*:*TypeUtils*'
GRPC_VERBOSITY=ERROR ./cmake_build/test/testing-it --gtest_filter='MilvusMockedTest.*:UnconnectMilvusMockedTest.*'
```

- Mocked tests use `test/it/mocks/MilvusMockedTest` fixtures (`MilvusMockedTest`, `UnconnectMilvusMockedTest`)
  and a `StrictMock<MilvusMockedService>`; assert the proto request contents that actually go on the wire.
- Windows CI intentionally skips `testing-it` because the in-process mocked server is unstable across the
  EXE/DLL boundary.
- System tests start a Milvus container via `test/st/milvus_container.py`; obtain permission before running
  them (`make st`) and clean up resources they create.

## Examples / tutorials

- Prefer `examples/src/v2/` for new examples, matching the V2-first policy. Examples mutate Milvus (create,
  insert, search, drop); review connection settings and clean up resources.
- `tutorial/` apps pin the published SDK version in their build; keep tutorial code and the pinned version
  mutually compatible in the same commit.

## CI and coverage

- `.github/workflows/main.yaml` gates on a `changes` filter (docs/meta files skip CI), then runs:
  - `linux` matrix: Ubuntu 20.04 (gcc 9) and Fedora 39, both with lint (Ubuntu), unit/mocked tests, tutorials
    (Ubuntu), and package verification;
  - `coverage-ubuntu` (Ubuntu 22.04, lcov + codecov);
  - `macos-15` and `windows-2022`.
- Conan and ccache are cached per job (keys scoped by branch and source hash); `scripts/prune_caches.py`
  prunes old generations. Push-to-master runs warm the default-branch caches for subsequent PRs.
- Coverage uploads `code_coverage/lcov_output.info` with `codecov/codecov-action@v5`; generated protobuf code
  is excluded from the report.

## Git / PR conventions

- The PR branch must contain exactly one commit; the commit message must carry a `Signed-off-by` trailer.
  Squash new work into the single PR commit and force-push with `git commit --amend -sm '...'` /
  `git push --force-with-lease`.
- Primary remote for upstream is `source` (`milvus-io/milvus-sdk-cpp`); `origin` is the contributor fork.
