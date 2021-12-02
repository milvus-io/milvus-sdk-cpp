#include "MilvusMockedServer.h"

#include <grpc++/server.h>
#include <grpc++/server_builder.h>

milvus::MilvusMockedServer::MilvusMockedServer(milvus::MilvusMockedService& service) : service_{service} {
}

uint16_t
milvus::MilvusMockedServer::ListenPort() const {
    return static_cast<uint16_t>(listen_port_);
}

void
milvus::MilvusMockedServer::Start() {
    ::grpc::ServerBuilder builder;
    builder.AddListeningPort("[::]:0", ::grpc::InsecureServerCredentials(), &listen_port_);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
}

void
milvus::MilvusMockedServer::Stop() {
    if (server_) {
        server_->Shutdown();
    }
}