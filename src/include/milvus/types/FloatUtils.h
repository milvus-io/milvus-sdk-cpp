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

namespace Eigen {
struct half;
struct bfloat16;
}  // namespace Eigen

namespace milvus {

template <typename Fp16T, typename FloatT>
std::vector<FloatT>
Float16NumVecBytesToFloatNumVec(const std::string& val);

template <typename FloatT, typename Fp16T>
std::string
FloatNumVecToFloat16NumVecBytes(const std::vector<FloatT>& data);

extern template std::vector<Eigen::half>
Float16NumVecBytesToFloatNumVec<Eigen::half, Eigen::half>(const std::string& val);
extern template std::vector<float>
Float16NumVecBytesToFloatNumVec<Eigen::half, float>(const std::string& val);
extern template std::vector<double>
Float16NumVecBytesToFloatNumVec<Eigen::half, double>(const std::string& val);

extern template std::vector<Eigen::bfloat16>
Float16NumVecBytesToFloatNumVec<Eigen::bfloat16, Eigen::bfloat16>(const std::string& val);
extern template std::vector<float>
Float16NumVecBytesToFloatNumVec<Eigen::bfloat16, float>(const std::string& val);
extern template std::vector<double>
Float16NumVecBytesToFloatNumVec<Eigen::bfloat16, double>(const std::string& val);

extern template std::string
FloatNumVecToFloat16NumVecBytes<Eigen::half, Eigen::half>(const std::vector<Eigen::half>& data);
extern template std::string
FloatNumVecToFloat16NumVecBytes<float, Eigen::half>(const std::vector<float>& data);
extern template std::string
FloatNumVecToFloat16NumVecBytes<double, Eigen::half>(const std::vector<double>& data);

extern template std::string
FloatNumVecToFloat16NumVecBytes<Eigen::bfloat16, Eigen::bfloat16>(const std::vector<Eigen::bfloat16>& data);
extern template std::string
FloatNumVecToFloat16NumVecBytes<float, Eigen::bfloat16>(const std::vector<float>& data);
extern template std::string
FloatNumVecToFloat16NumVecBytes<double, Eigen::bfloat16>(const std::vector<double>& data);

}  // namespace milvus
