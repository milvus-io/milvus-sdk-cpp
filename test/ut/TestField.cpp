#include <gmock/gmock.h>

#include "param/Field.h"

class FieldTest : public ::testing::Test {};

using ::milvus::DataType;
using ::milvus::param::Field;
using ::testing::ElementsAreArray;

TEST_F(FieldTest, TestBoolField) {
    std::vector<bool> data{true, false, false};
    Field<bool> field{"foo", data};
    EXPECT_EQ(field.Type(), DataType::BOOL);
    EXPECT_EQ(field.Name(), "foo");
    EXPECT_THAT(field.Values(), ElementsAreArray(data));
}

TEST_F(FieldTest, TestInt8Field) {
    std::vector<int8_t> data{1, 2, 3};
    Field<int8_t> field{"foo", data};
    EXPECT_EQ(field.Type(), DataType::INT8);
    EXPECT_EQ(field.Name(), "foo");
    EXPECT_THAT(field.Values(), ElementsAreArray(data));
}

TEST_F(FieldTest, TestInt16Field) {
    std::vector<int16_t> data{1, 2, 3};
    Field<int16_t> field{"foo", data};
    EXPECT_EQ(field.Type(), DataType::INT16);
    EXPECT_EQ(field.Name(), "foo");
    EXPECT_THAT(field.Values(), ElementsAreArray(data));
}

TEST_F(FieldTest, TestInt32Field) {
    std::vector<int32_t> data{1, 2, 3};
    Field<int32_t> field{"foo", data};
    EXPECT_EQ(field.Type(), DataType::INT32);
    EXPECT_EQ(field.Name(), "foo");
    EXPECT_THAT(field.Values(), ElementsAreArray(data));
}

TEST_F(FieldTest, TestInt64Field) {
    std::vector<int64_t> data{1, 2, 3};
    Field<int64_t> field{"foo", data};
    EXPECT_EQ(field.Type(), DataType::INT64);
    EXPECT_EQ(field.Name(), "foo");
    EXPECT_THAT(field.Values(), ElementsAreArray(data));
}

TEST_F(FieldTest, TestFloatField) {
    std::vector<float> data{1.0f, 2.0f, 3.0f};
    Field<float> field{"foo", data};
    EXPECT_EQ(field.Type(), DataType::FLOAT);
    EXPECT_EQ(field.Name(), "foo");
    EXPECT_THAT(field.Values(), ElementsAreArray(data));
}

TEST_F(FieldTest, TestDoubleField) {
    std::vector<double> data{1.0, 2.0, 3.0};
    Field<double> field{"foo", data};
    EXPECT_EQ(field.Type(), DataType::DOUBLE);
    EXPECT_EQ(field.Name(), "foo");
    EXPECT_THAT(field.Values(), ElementsAreArray(data));
}

TEST_F(FieldTest, TestStringField) {
    std::vector<std::string> data{"abc", "xyz", ""};
    Field<std::string> field{"foo", data};
    EXPECT_EQ(field.Type(), DataType::STRING);
    EXPECT_EQ(field.Name(), "foo");
    EXPECT_THAT(field.Values(), ElementsAreArray(data));
}

TEST_F(FieldTest, TestFloatsField) {
    std::vector<std::vector<float>> data{{1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 3.0f}};
    Field<std::vector<float>> field{"foo", data};
    EXPECT_EQ(field.Type(), DataType::FLOAT_VECTOR);
    EXPECT_EQ(field.Name(), "foo");
    EXPECT_THAT(field.Values(), ElementsAreArray(data));
}

TEST_F(FieldTest, TestBytesField) {
    std::vector<std::vector<char>> data{{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
    Field<std::vector<char>> field{"foo", data};
    EXPECT_EQ(field.Type(), DataType::BINARY_VECTOR);
    EXPECT_EQ(field.Name(), "foo");
    EXPECT_THAT(field.Values(), ElementsAreArray(data));
}