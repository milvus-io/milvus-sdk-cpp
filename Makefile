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

all-debug: build-sdk-debug
all-release: build-sdk-release
all: all-debug

# Code lint
lint:
	@(env bash ${PWD}/scripts/build.sh -l $(GRPC_PATH))

# Build sdk
build-sdk-debug:
	@echo "Building Milvus SDK debug version ..."
	@(env bash $(PWD)/scripts/build.sh -t Debug $(GRPC_PATH))

build-sdk-release:
	@echo "Building Milvus SDK release version ..."
	@(env bash $(PWD)/scripts/build.sh -t Release $(GRPC_PATH))

install: install-release

install-release:
	@echo "Installing Milvus SDK release version ..."
	@(env bash $(PWD)/scripts/build.sh -i -t Release $(GRPC_PATH))

install-debug:
	@echo "Installing Milvus SDK debug version ..."
	@(env bash $(PWD)/scripts/build.sh -i -t Debug $(GRPC_PATH))

test:
	@echo "Testing with Milvus SDK"
	@(env bash $(PWD)/scripts/build.sh -u $(GRPC_PATH))

st:
	@echo "System Testing with Milvus SDK"
	@(env bash $(PWD)/scripts/build.sh -s $(GRPC_PATH))

coverage:
	@echo "Run code coverage ..."
	@(env bash $(PWD)/scripts/build.sh -u -c $(GRPC_PATH))
	@(env bash $(PWD)/scripts/coverage.sh)

doc:
	@echo "Generating Milvus SDK documentation ..."
	rm -rf ./doc/html ./doc/latex
	doxygen ./doc/Doxyfile

clean:
	@echo "Cleaning"
	rm -fr cmake_build/

.PHONY: test clean doc
