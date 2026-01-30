#include <gtest/gtest.h>

#include <array>
#include <cstdlib>

#include "milvus/MilvusClient.h"
#include "milvus/Status.h"
#ifdef MILVUS_WITH_TESTCONTAINERS
#include "TestcontainersMilvus.h"
#endif

int
main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

#ifdef MILVUS_WITH_TESTCONTAINERS
    bool enable_testcontainers = true;
    const char* testcontainers_env = std::getenv("MILVUS_TESTCONTAINERS");
    if (testcontainers_env && testcontainers_env[0] == '0') {
        enable_testcontainers = false;
    }
    ::testing::AddGlobalTestEnvironment(new milvus::test::MilvusTestcontainersEnvironment(enable_testcontainers));
#endif

    std::cout << "======== Test with milvus server ========" << std::endl;
    ::testing::GTEST_FLAG(filter) = "-MilvusServerTestWithAuth.*";
    int result = RUN_ALL_TESTS();
    if (result > 0) {
        return result;
    }

    return 0;
}
