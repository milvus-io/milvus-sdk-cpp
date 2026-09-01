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

PWD 	:= $(shell pwd)
V2_EXAMPLES := $(basename $(notdir $(wildcard examples/src/v2/*.cpp)))
RUN_EXAMPLE := $(word 2,$(MAKECMDGOALS))
TUTORIAL_DIRS := $(sort $(wildcard tutorial/[0-9]_*))
TUTORIAL_NAMES := $(notdir $(TUTORIAL_DIRS))
RUN_TUTORIAL := $(word 2,$(MAKECMDGOALS))

ifeq ($(firstword $(MAKECMDGOALS)),run)
ifneq ($(RUN_EXAMPLE),)
.PHONY: $(RUN_EXAMPLE)
$(RUN_EXAMPLE):
	@:
endif
endif

ifeq ($(firstword $(MAKECMDGOALS)),run-tutorial)
ifneq ($(RUN_TUTORIAL),)
.PHONY: $(RUN_TUTORIAL)
$(RUN_TUTORIAL):
	@:
endif
endif

all-debug: build-sdk-debug
all-release: build-sdk-release
all: all-debug

# Code lint
lint:
	@(env bash ${PWD}/scripts/build.sh -l)

lint-release:
	@(env bash ${PWD}/scripts/build.sh -l -t Release)

# Build sdk
build-sdk-debug:
	@echo "Building Milvus SDK debug version ..."
	@(env bash $(PWD)/scripts/build.sh -t Debug)

build-sdk-release:
	@echo "Building Milvus SDK release version ..."
	@(env bash $(PWD)/scripts/build.sh -t Release)

test-release:
	@echo "Testing Milvus SDK release version ..."
	@(env bash $(PWD)/scripts/build.sh -u -t Release)

build-no-conan-debug:
	@echo "Building Milvus SDK debug version ..."
	@(env bash $(PWD)/scripts/build.sh -z -t Debug)

build-no-conan-release:
	@echo "Building Milvus SDK release version ..."
	@(env bash $(PWD)/scripts/build.sh -z -t Release)

install: install-release

install-release:
	@echo "Installing Milvus SDK release version ..."
	@(env bash $(PWD)/scripts/build.sh -i -t Release)

install-debug:
	@echo "Installing Milvus SDK debug version ..."
	@(env bash $(PWD)/scripts/build.sh -i -t Debug)

test:
	@echo "Testing with Milvus SDK"
	@(env bash $(PWD)/scripts/build.sh -u)

test-no-conan:
	@echo "Testing with Milvus SDK"
	@(env bash $(PWD)/scripts/build.sh -z -u)

# Configure and compile every standalone tutorial project.
tutorials:
	@(env JOBS=$(JOBS) bash $(PWD)/scripts/build_tutorials.sh)

# Run one built tutorial from the repository root, e.g. `make run-tutorial quickstart`.
run-tutorial:
	@set -eu; \
	tutorial="$(RUN_TUTORIAL)"; \
	if [ -z "$$tutorial" ] || [ "$(words $(MAKECMDGOALS))" -ne 2 ]; then \
		echo "Usage: make run-tutorial <tutorial>" >&2; \
		echo "Available tutorials: $(TUTORIAL_NAMES)" >&2; \
		exit 2; \
	fi; \
	dir="tutorial/$$tutorial"; \
	if [ ! -d "$$dir" ]; then \
		set -- tutorial/[0-9]_"$$tutorial"; \
		if [ ! -d "$$1" ]; then \
			echo "Unknown tutorial: $$tutorial" >&2; \
			echo "Available tutorials: $(TUTORIAL_NAMES)" >&2; \
			exit 2; \
		fi; \
		dir="$$1"; \
	fi; \
	if [ ! -d "$$dir/cmake_build" ]; then \
		echo "Tutorial is not built: $$dir" >&2; \
		echo "Build all tutorials first with: make tutorials" >&2; \
		exit 1; \
	fi; \
	$(MAKE) -C "$$dir" run

# Run one built V2 example by its source basename, e.g. `make run simple`.
run:
	@set -eu; \
	example="$(RUN_EXAMPLE)"; \
	if [ -z "$$example" ] || [ "$(words $(MAKECMDGOALS))" -ne 2 ]; then \
		echo "Usage: make run <v2-example>" >&2; \
		echo "Available V2 examples: $(V2_EXAMPLES)" >&2; \
		exit 2; \
	fi; \
	case " $(V2_EXAMPLES) " in \
		*" $$example "*) ;; \
		*) echo "Unknown V2 example: $$example" >&2; \
		   echo "Available V2 examples: $(V2_EXAMPLES)" >&2; \
		   exit 2 ;; \
	esac; \
	binary="$(PWD)/cmake_build/examples/v2/sdk_$${example}_v2"; \
	if [ ! -x "$$binary" ]; then \
		echo "Example is not built: $$binary" >&2; \
		echo "Build it first with: cmake --build cmake_build --target sdk_$${example}_v2" >&2; \
		exit 1; \
	fi; \
	GRPC_VERBOSITY=$${GRPC_VERBOSITY:-ERROR} "$$binary"

st:
	@echo "System Testing with Milvus SDK"
	@(env bash $(PWD)/scripts/build.sh -s)

coverage:
	@echo "Run code coverage ..."
	@(env bash $(PWD)/scripts/build.sh -u -s -c)
	@(env bash $(PWD)/scripts/coverage.sh)

doc:
	@echo "Generating Milvus SDK documentation ..."
	rm -rf ./doc/html ./doc/latex
	doxygen ./doc/Doxyfile

package:
	@echo "Packaging Milvus SDK release version ..."
	@# Release builds must be reproducible and must not mutate the source
	@# tree. Invoke build.sh directly with -f to skip the in-place
	@# clang-format that build-sdk-release would otherwise apply.
	@(env bash $(PWD)/scripts/build.sh -f -t Release)
	@(cd cmake_build && cpack)

clean:
	@echo "Cleaning"
	rm -fr cmake_build/ build/

.PHONY: lint lint-release test tutorials run-tutorial clean doc package run
