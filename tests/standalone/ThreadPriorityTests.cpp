#include <algorithm>
#include <thread>

#include <sys/resource.h>

#include <gtest/gtest.h>

#include <spacetime/interface/standalone/ThreadPriority.hpp>

namespace Spacetime::Standalone
{
    TEST(ThreadPriority, LowersOnlyTheCallingThread)
    {
        const int main_nice = getpriority(PRIO_PROCESS, 0);
        int before = 0, after = 0;
        std::thread worker([&]()
                           {
            before = getpriority(PRIO_PROCESS, 0);
            lowerCurrentThreadPriority(3);
            after = getpriority(PRIO_PROCESS, 0); });
        worker.join();

        EXPECT_EQ(after, std::min(before + 3, 19)); // nice values are capped at 19
        EXPECT_EQ(getpriority(PRIO_PROCESS, 0), main_nice);
    }
} // namespace Spacetime::Standalone
