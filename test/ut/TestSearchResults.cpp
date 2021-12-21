#include <gtest/gtest.h>

#include "types/SearchResults.h"

class SearchResultsTest : public ::testing::Test {};

TEST_F(SearchResultsTest, GeneralTesting) {
    std::vector<milvus::SingleResult> result_array = {milvus::SingleResult()};

    milvus::SearchResults results(result_array);
    EXPECT_EQ(1, results.Results().size());
}
