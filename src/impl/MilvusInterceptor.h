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

#include <grpcpp/grpcpp.h>
#include <grpcpp/support/client_interceptor.h>

#include <string>
#include <unordered_map>

class HeaderAdderInterceptor : public grpc::experimental::Interceptor {
 public:
    explicit HeaderAdderInterceptor(const std::unordered_map<std::string, std::string>& headers);

    void
    Intercept(grpc::experimental::InterceptorBatchMethods* methods) override;

 private:
    std::unordered_map<std::string, std::string> headers_;
};

class HeaderAdderInterceptorFactory : public grpc::experimental::ClientInterceptorFactoryInterface {
 public:
    explicit HeaderAdderInterceptorFactory(const std::unordered_map<std::string, std::string>& metadata);

    grpc::experimental::Interceptor*
    CreateClientInterceptor(grpc::experimental::ClientRpcInfo* info) override;

 private:
    std::unordered_map<std::string, std::string> metadata_;
};

std::shared_ptr<grpc::Channel>
CreateChannelWithHeaderInterceptor(const std::string& target, const std::shared_ptr<grpc::ChannelCredentials>& creds,
                                   const grpc::ChannelArguments& args,
                                   const std::unordered_map<std::string, std::string>& metadata);
