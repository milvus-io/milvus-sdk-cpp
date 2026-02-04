// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * Google Test Environment that manages Milvus container lifecycle.
 *
 * - SetUp(): Starts Milvus container before any tests run
 * - TearDown(): Stops the container after all tests complete
 */
class MilvusContainerEnv : public ::testing::Environment {
 public:
    void
    SetUp() override {
        std::string py_version = execCommand("python3 --version 2>&1 | cut -d' ' -f2");
        py_version.erase(py_version.find_last_not_of(" \n\r\t") + 1);
        std::cout << "python3 version: " << py_version << std::endl;
        std::cout << "Starting Milvus container..." << std::endl;

        std::string script_path = findScript();
        std::string cmd = "python3 " + script_path + " start 2>&1";

        container_id_ = execCommand(cmd);
        std::cout << "execCommand returns: " << container_id_ << std::endl;

        // Trim whitespace/newlines from container ID
        container_id_.erase(container_id_.find_last_not_of(" \n\r\t") + 1);
        container_id_.erase(0, container_id_.find_first_not_of(" \n\r\t"));

        if (container_id_.empty() || container_id_.length() < 12) {
            throw std::runtime_error("Failed to start Milvus container. Output: " + container_id_);
        }

        std::cout << "Milvus container started: " << container_id_ << std::endl;

        // Get container IP for test connections (localhost may not work in CI)
        std::string inspect_cmd = "docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' " +
                                  container_id_ + " 2>&1";
        std::string container_ip = execCommand(inspect_cmd);
        container_ip.erase(container_ip.find_last_not_of(" \n\r\t") + 1);
        container_ip.erase(0, container_ip.find_first_not_of(" \n\r\t"));
        std::cout << "Milvus container IP: " << container_ip << std::endl;

        setenv("MILVUS_HOST", container_ip.c_str(), 1);
    }

    void
    TearDown() override {
        if (!container_id_.empty()) {
            std::cout << "Stopping Milvus container: " << container_id_.substr(0, 12) << std::endl;

            std::string script_path = findScript();
            std::string cmd = "python3 " + script_path + " stop " + container_id_ + " 2>&1";

            execCommand(cmd);
            std::cout << "Milvus container stopped" << std::endl;
        }
    }

 private:
    std::string container_id_;

    std::string
    execCommand(const std::string& cmd) {
        std::array<char, 256> buffer;
        std::string result;

        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
        if (!pipe) {
            throw std::runtime_error("popen() failed");
        }

        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }

        return result;
    }

    std::string
    findScript() {
        // Try common locations for the script
        const char* env_path = std::getenv("MILVUS_CONTAINER_SCRIPT");
        if (env_path != nullptr) {
            return {env_path};
        }

        // Default: assume script is in the same directory as the test binary
        // or in the source tree location
        std::vector<std::string> paths = {
            "milvus_container.py",
            "test/st/milvus_container.py",
            "../test/st/milvus_container.py",
        };

        for (const auto& path : paths) {
            FILE* f = fopen(path.c_str(), "r");
            if (f != nullptr) {
                fclose(f);
                return path;
            }
        }

        // Fallback
        return "test/st/milvus_container.py";
    }
};
