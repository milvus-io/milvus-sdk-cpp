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
#include <cctype>
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
    static std::string
    Trim(std::string value) {
        auto start = value.find_first_not_of(" \n\r\t");
        if (start == std::string::npos) {
            return "";
        }
        auto end = value.find_last_not_of(" \n\r\t");
        return value.substr(start, end - start + 1);
    }

    static bool
    IsHexContainerId(const std::string& value) {
        if (value.length() < 12) {
            return false;
        }
        for (char c : value) {
            if (!std::isxdigit(static_cast<unsigned char>(c))) {
                return false;
            }
        }
        return true;
    }

    static std::string
    ExtractContainerId(const std::string& output) {
        size_t pos = 0;
        std::string last_match;
        while (pos <= output.size()) {
            auto end = output.find('\n', pos);
            std::string line = end == std::string::npos ? output.substr(pos) : output.substr(pos, end - pos);
            line = Trim(line);
            if (IsHexContainerId(line)) {
                last_match = line;
            }
            if (end == std::string::npos) {
                break;
            }
            pos = end + 1;
        }
        return last_match;
    }

    void
    SetUp() override {
        std::string py_version = execCommand("python3 --version 2>&1 | cut -d' ' -f2");
        std::cout << "python3 version: " << py_version << std::endl;
        std::cout << "Starting Milvus container..." << std::endl;

        std::string script_path = findScript();
        std::string cmd = "python3 " + script_path + " start 2>&1";

        std::string start_output = execCommand(cmd);
        std::cout << "execCommand returns: " << start_output << std::endl;

        container_id_ = ExtractContainerId(start_output);
        if (container_id_.empty()) {
            throw std::runtime_error("Failed to start Milvus container. Output: " + start_output);
        }
        std::cout << "Milvus container started: " << container_id_ << std::endl;

        // Get container IP for test connections (localhost may not work in CI)
        std::string inspect_cmd =
            "docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' " + container_id_ + " 2>&1";
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
