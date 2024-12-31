#include "MilvusInterceptor.h"

HeaderAdderInterceptor::HeaderAdderInterceptor(const std::vector<std::pair<std::string, std::string>>& headers)
    : headers_(headers) {}

void HeaderAdderInterceptor::Intercept(grpc::experimental::InterceptorBatchMethods* methods) {
    if (methods->QueryInterceptionHookPoint(
            grpc::experimental::InterceptionHookPoints::PRE_SEND_INITIAL_METADATA)) {
        for (const auto& header : headers_) {
            methods->GetSendInitialMetadata()->insert({header.first, header.second});
        }
    }
    methods->Proceed();
}

HeaderAdderInterceptorFactory::HeaderAdderInterceptorFactory(const std::vector<std::pair<std::string, std::string>>& headers)
    : headers_(headers) {}

grpc::experimental::Interceptor* HeaderAdderInterceptorFactory::CreateClientInterceptor(
    grpc::experimental::ClientRpcInfo* info) {
    return new HeaderAdderInterceptor(headers_);
}

std::shared_ptr<grpc::Channel> CreateChannelWithHeaderInterceptor(
    const std::string& target,
    const std::shared_ptr<grpc::ChannelCredentials>& creds,
    const grpc::ChannelArguments& args,
    const std::vector<std::pair<std::string, std::string>>& headers) {

    std::vector<std::unique_ptr<grpc::experimental::ClientInterceptorFactoryInterface>> interceptor_creators;
    interceptor_creators.push_back(std::make_unique<HeaderAdderInterceptorFactory>(headers));

    return grpc::experimental::CreateCustomChannelWithInterceptors(target, creds, args, std::move(interceptor_creators));
}
