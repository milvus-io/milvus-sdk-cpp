#include <gtest/gtest.h>

#include <limits>

#include "types/ProgressMonitor.h"

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
