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

#include <gtest/gtest.h>

#include <limits>

#include "milvus/types/ProgressMonitor.h"

class ProgressMonitorTest : public ::testing::Test {};

TEST_F(ProgressMonitorTest, DefaultSetting) {
    milvus::ProgressMonitor pm;
    EXPECT_EQ(pm.CheckTimeout(), 60);
    EXPECT_EQ(pm.CheckInterval(), 500);

    auto no_wait = milvus::ProgressMonitor::NoWait();
    EXPECT_EQ(no_wait.CheckTimeout(), 0);

    auto forever = milvus::ProgressMonitor::Forever();
    EXPECT_EQ(forever.CheckTimeout(), std::numeric_limits<uint32_t>::max());
}

TEST_F(ProgressMonitorTest, Setting) {
    milvus::ProgressMonitor pm{100};
    EXPECT_EQ(pm.CheckTimeout(), 100);

    pm.SetCheckInterval(100);
    EXPECT_EQ(pm.CheckInterval(), 100);
}

TEST_F(ProgressMonitorTest, Callback) {
    milvus::Progress progress(50, 100);
    auto func = [&](milvus::Progress& p) -> void {
        EXPECT_EQ(p.finished_, progress.finished_);
        EXPECT_EQ(p.total_, progress.total_);
    };

    milvus::ProgressMonitor pm;
    pm.SetCallbackFunc(func);
    pm.DoProgress(progress);
}
