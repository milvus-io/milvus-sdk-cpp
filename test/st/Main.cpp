#include <gtest/gtest.h>

#include <array>

#include "milvus/MilvusClient.h"
#include "milvus/Status.h"

int
main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "======== Test with milvus server ========" << std::endl;
    ::testing::GTEST_FLAG(filter) = "-MilvusServerTestWithAuth.*";
    int result = RUN_ALL_TESTS();
    if (result > 0) {
        return result;
    }

    return 0;
}
