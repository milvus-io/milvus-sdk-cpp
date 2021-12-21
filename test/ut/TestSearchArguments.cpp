#include <gtest/gtest.h>

#include "types/SearchArguments.h"

class SearchArgumentsTest : public ::testing::Test {};

TEST_F(SearchArgumentsTest, GeneralTesting) {
    milvus::SearchArguments arguments;

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

    std::string expression = "expr";
    arguments.SetExpression(expression);
    EXPECT_EQ(expression, arguments.Expression());
    EXPECT_TRUE(arguments.SetExpression(empty_name).IsOk());

    uint64_t ts = 1000;
    arguments.SetTravelTimestamp(ts);
    EXPECT_EQ(ts, arguments.TravelTimestamp());
    arguments.SetGuaranteeTimestamp(ts);
    EXPECT_EQ(ts, arguments.GuaranteeTimestamp());
}

TEST_F(SearchArgumentsTest, VectorTesting) {
    milvus::BinaryVecFieldData::ElementT binary_vector = {1, 2, 3};
    milvus::FloatVecFieldData::ElementT float_vector = {1.0, 2.0};

    {
        milvus::SearchArguments arguments;
        auto status = arguments.AddTargetVector(binary_vector);
        EXPECT_TRUE(status.IsOk());

        status = arguments.AddTargetVector(float_vector);
        EXPECT_FALSE(status.IsOk());

        milvus::BinaryVecFieldData::ElementT new_vector = {1, 2};
        status = arguments.AddTargetVector(new_vector);
        EXPECT_FALSE(status.IsOk());

        auto target_vectors = arguments.TargetVectors();
        EXPECT_TRUE(target_vectors != nullptr);
        EXPECT_EQ(milvus::DataType::BINARY_VECTOR, target_vectors->Type());
        EXPECT_EQ(1, target_vectors->Count());
    }

    {
        milvus::SearchArguments arguments;
        auto status = arguments.AddTargetVector(float_vector);
        EXPECT_TRUE(status.IsOk());

        status = arguments.AddTargetVector(binary_vector);
        EXPECT_FALSE(status.IsOk());

        milvus::FloatVecFieldData::ElementT new_vector = {1.0, 2.0, 3.0};
        status = arguments.AddTargetVector(new_vector);
        EXPECT_FALSE(status.IsOk());

        auto target_vectors = arguments.TargetVectors();
        EXPECT_TRUE(target_vectors != nullptr);
        EXPECT_EQ(milvus::DataType::FLOAT_VECTOR, target_vectors->Type());
        EXPECT_EQ(1, target_vectors->Count());
    }
}
