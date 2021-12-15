#include <gtest/gtest.h>

#include "types/IDArray.h"

class IDArrayTest : public ::testing::Test {};

TEST_F(IDArrayTest, GeneralTesting) {
    {
        std::vector<int64_t> ids = {1, 2, 3};
        milvus::IDArray id_array(ids);
        EXPECT_TRUE(id_array.IsIntegerID());
        EXPECT_EQ(id_array.IntIDArray().size(), ids.size());
        EXPECT_TRUE(id_array.StrIDArray().empty());
    }

    {
        std::vector<std::string> ids = {"a", "b", "c"};
        milvus::IDArray id_array(ids);
        EXPECT_FALSE(id_array.IsIntegerID());
        EXPECT_EQ(id_array.StrIDArray().size(), ids.size());
        EXPECT_TRUE(id_array.IntIDArray().empty());
    }
}
