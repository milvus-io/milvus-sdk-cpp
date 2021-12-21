#include <gtest/gtest.h>

#include "types/QueryArguments.h"

class QueryArgumentsTest : public ::testing::Test {};

TEST_F(QueryArgumentsTest, GeneralTesting) {
    milvus::QueryArguments arguments;

    std::string empty_name = "";
    std::string collection_name = "test";
    arguments.SetCollectionName(collection_name);
    EXPECT_FALSE(arguments.SetCollectionName(empty_name).IsOk());
    EXPECT_EQ(collection_name, arguments.CollectionName());

    std::string partition_name = "p1";
    arguments.AddPartitionName(partition_name);
    EXPECT_FALSE(arguments.AddPartitionName(empty_name).IsOk());
    EXPECT_EQ(1, arguments.PartitionNames().size());
    auto names = arguments.PartitionNames();
    EXPECT_TRUE(names.find(partition_name) != names.end());

    std::string field_name = "f1";
    arguments.AddOutputField(field_name);
    EXPECT_FALSE(arguments.AddOutputField(empty_name).IsOk());
    EXPECT_EQ(1, arguments.OutputFields().size());
    auto field_names = arguments.OutputFields();
    EXPECT_TRUE(field_names.find(field_name) != field_names.end());

    std::string expression = "expr";
    arguments.SetExpression(expression);
    EXPECT_FALSE(arguments.SetExpression(empty_name).IsOk());
    EXPECT_EQ(expression, arguments.Expression());

    uint64_t ts = 1000;
    arguments.SetTravelTimestamp(ts);
    EXPECT_EQ(ts, arguments.TravelTimestamp());
    arguments.SetGuaranteeTimestamp(ts);
    EXPECT_EQ(ts, arguments.GuaranteeTimestamp());
}
