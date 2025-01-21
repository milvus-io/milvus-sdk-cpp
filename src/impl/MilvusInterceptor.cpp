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

#include "MilvusInterceptor.h"

HeaderAdderInterceptor::HeaderAdderInterceptor(const std::vector<std::pair<std::string, std::string>>& headers)
    : headers_(headers) {
}

void
HeaderAdderInterceptor::Intercept(grpc::experimental::InterceptorBatchMethods* methods) {
    if (methods->QueryInterceptionHookPoint(grpc::experimental::InterceptionHookPoints::PRE_SEND_INITIAL_METADATA)) {
        for (const auto& header : headers_) {
            methods->GetSendInitialMetadata()->insert({header.first, header.second});
        }
    }
    methods->Proceed();
}

HeaderAdderInterceptorFactory::HeaderAdderInterceptorFactory(
    const std::vector<std::pair<std::string, std::string>>& headers)
    : headers_(headers) {
}

grpc::experimental::Interceptor*
HeaderAdderInterceptorFactory::CreateClientInterceptor(grpc::experimental::ClientRpcInfo* info) {
    return new HeaderAdderInterceptor(headers_);
}

std::shared_ptr<grpc::Channel>
CreateChannelWithHeaderInterceptor(const std::string& target, const std::shared_ptr<grpc::ChannelCredentials>& creds,
                                   const grpc::ChannelArguments& args,
                                   const std::vector<std::pair<std::string, std::string>>& headers) {
    std::vector<std::unique_ptr<grpc::experimental::ClientInterceptorFactoryInterface>> interceptor_creators;
    interceptor_creators.push_back(std::make_unique<HeaderAdderInterceptorFactory>(headers));

    return grpc::experimental::CreateCustomChannelWithInterceptors(target, creds, args,
                                                                   std::move(interceptor_creators));
}
