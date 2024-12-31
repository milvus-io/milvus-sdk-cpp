#pragma once

#include <grpcpp/grpcpp.h>
#include <grpcpp/support/client_interceptor.h>
#include <vector>
#include <string>

class HeaderAdderInterceptor : public grpc::experimental::Interceptor {
public:
    HeaderAdderInterceptor(const std::vector<std::pair<std::string, std::string>>& headers);

    void Intercept(grpc::experimental::InterceptorBatchMethods* methods) override;

private:
    std::vector<std::pair<std::string, std::string>> headers_;
};

class HeaderAdderInterceptorFactory : public grpc::experimental::ClientInterceptorFactoryInterface {
public:
    HeaderAdderInterceptorFactory(const std::vector<std::pair<std::string, std::string>>& headers);

    grpc::experimental::Interceptor* CreateClientInterceptor(
        grpc::experimental::ClientRpcInfo* info) override;

private:
    std::vector<std::pair<std::string, std::string>> headers_;
};

std::shared_ptr<grpc::Channel> CreateChannelWithHeaderInterceptor(
    const std::string& target,
    const std::shared_ptr<grpc::ChannelCredentials>& creds,
    const grpc::ChannelArguments& args,
    const std::vector<std::pair<std::string, std::string>>& headers);
