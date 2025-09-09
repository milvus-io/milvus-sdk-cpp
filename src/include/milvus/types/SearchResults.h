// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <vector>

#include "FieldData.h"
#include "IDArray.h"

namespace milvus {

/**
 * @brief Topk results for one target vector of MilvusClient::Search()
 */
struct SingleResult {
    SingleResult() = default;

    /**
     * @brief Constructor
     */
    SingleResult(const SingleResult& src);

    /**
     * @brief Constructor
     * Note: this constructor might throw exception in the cases of:
     *   1. the pk_name or score_name is empty
     *   2. row count of fields are unequal
     */
    SingleResult(const std::string& pk_name, const std::string& score_name, std::vector<FieldDataPtr>&& output_fields);

    /**
     * @brief Distances/scores array of one target vector
     */
    const std::vector<float>&
    Scores() const;

    /**
     * @brief Topk id array of one target vector
     * Note: the returned IDArray is a temporary object copied from FieldData. It is recommended to
     * use OutputField() method like this:
     *    FieldDataPtr ids = result.OutputField(result.PrimaryKeyName());
     */
    IDArray
    Ids() const;

    /**
     * @brief The primary key name
     * Sometimes the caller of Search() doesn't know the pk name, the server returns this name
     * so that you don't need to describe the collection again.
     */
    const std::string&
    PrimaryKeyName() const;

    /**
     * @brief Score field name in search result
     * Note: the default score name is "score", but if your collection schema already has a "score" field,
     * and the "score" field is an output field, the score name will be changed to "_score". If "_socre" is
     * also duplicated, then the score name will be changed to "__score", etc.
     */
    const std::string&
    ScoreName() const;

    /**
     * @brief Output fields data
     */
    const std::vector<FieldDataPtr>&
    OutputFields() const;

    /**
     * @brief Get an output field by name
     */
    FieldDataPtr
    OutputField(const std::string& name) const;

    /**
     * @brief Get an output field by name and cast to specific pointer
     */
    template <typename T>
    std::shared_ptr<T>
    OutputField(const std::string& name) const {
        return std::dynamic_pointer_cast<T>(OutputField(name));
    }

    /**
     * @brief Get all output rows.
     */
    Status
    OutputRows(EntityRows& rows) const;

    /**
     * @brief Get row data. Throw exception if the i is out of bound.
     */
    Status
    OutputRow(int i, EntityRow& row) const;

    /**
     * @brief Get row count of the result.
     */
    uint64_t
    GetRowCount() const;

 private:
    void
    verify() const;

 private:
    std::string pk_name_;     // the server tells primary key name so that you don't need to describe the collection
    std::string score_name_;  // name of score field, default is "score". if duplicated, the name could be "_socre"
    std::vector<FieldDataPtr> output_fields_;
};

/**
 * @brief Results returned by MilvusClient::Search().
 */
class SearchResults {
 public:
    SearchResults();

    /**
     * @brief Constructor
     */
    SearchResults(const SearchResults& src);

    /**
     * @brief Constructor
     */
    explicit SearchResults(std::vector<SingleResult>&& results);

    /**
     * @brief Constructor
     */
    explicit SearchResults(const std::vector<SingleResult>& results);

    /**
     * @brief Get search results.
     */
    const std::vector<SingleResult>&
    Results() const;

 private:
    std::vector<SingleResult> nq_results_{};
};

}  // namespace milvus
