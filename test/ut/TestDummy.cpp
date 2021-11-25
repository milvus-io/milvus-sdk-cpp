#include <gtest/gtest.h>

class DummyTest: public ::testing::Test {

};

TEST_F(DummyTest, Foo) {
    EXPECT_EQ(0, 0);
}