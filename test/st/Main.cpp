#include <gtest/gtest.h>

#include <array>

#include "PythonMilvusServer.h"
#include "milvus/MilvusClient.h"
#include "milvus/Status.h"

inline void
waitMilvusServerReady(const milvus::test::PythonMilvusServer& server) {
    int max_retry = 60, retry = 0;
    bool has;

    auto client = milvus::MilvusClient::Create();
    auto param = server.TestClientParam();
    client->Connect(*param);
    auto status = client->HasCollection("no_such", has);

    while (!status.IsOk() && retry++ < max_retry) {
        std::this_thread::sleep_for(std::chrono::seconds{5});
        client = milvus::MilvusClient::Create();
        client->Connect(*param);
        status = client->HasCollection("no_such", has);
        std::cout << "Wait milvus start done, try: " << retry << ", status: " << status.Message() << std::endl;
    }
    std::cout << "Wait milvus start done, status: " << status.Message() << std::endl;
}

std::shared_ptr<milvus::ConnectParam> s_connectParam;

int
main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // tls mode
    {
        std::cout << "======== Test TLS mode ========" << std::endl;

        std::array<char, 256> path{};
        getcwd(path.data(), path.size());
        std::string pwd = path.data();

        milvus::test::PythonMilvusServer server_{};
        server_.SetTls(2, pwd + "/certs/server.crt", pwd + "/certs/server.key", pwd + "/certs/ca.crt");
        // server_.SetAuthorizationEnabled(true);

        server_.Start();
        waitMilvusServerReady(server_);
        s_connectParam = server_.TestClientParam();

        ::testing::GTEST_FLAG(filter) = "-MilvusServerTestWithAuth.*";
        int result = RUN_ALL_TESTS();
        if (result > 0) {
            return result;
        }
        server_.Stop();
    }

    // auth mode
    {
        std::cout << "======== Test Auth mode ========" << std::endl;

        milvus::test::PythonMilvusServer server_{};
        server_.SetAuthorizationEnabled(true);

        server_.Start();
        waitMilvusServerReady(server_);
        s_connectParam = server_.TestClientParam();

        ::testing::GTEST_FLAG(filter) = "-MilvusServerTestWithTlsMode.*";
        int result = RUN_ALL_TESTS();
        if (result > 0) {
            return result;
        }
        server_.Stop();
    }

    return 0;
}
