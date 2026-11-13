#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include <spacetime/interface/Dataloader.hpp>
#include <spacetime/interface/EndTime.hpp>

namespace Spacetime::Interface
{
    namespace
    {
        /// A dataloader with a fixed last time (nullopt: live).
        struct FixedDataloader : public Dataloader
        {
            explicit FixedDataloader(std::optional<double> last_time) : m_last_time(last_time) {}
            std::optional<double> lastTime() const override { return m_last_time; }
            std::optional<double> m_last_time;
        };

        std::vector<std::shared_ptr<Dataloader>> loaders(std::initializer_list<std::optional<double>> last_times)
        {
            std::vector<std::shared_ptr<Dataloader>> out;
            for (const auto &t : last_times)
                out.push_back(std::make_shared<FixedDataloader>(t));
            return out;
        }
    } // namespace

    TEST(EndTime, ANonNegativeEndTimeIsKept)
    {
        EXPECT_DOUBLE_EQ(resolveEndTime(3.0, loaders({1.0})), 3.0);
        EXPECT_DOUBLE_EQ(resolveEndTime(0.0, loaders({std::nullopt})), 0.0);
    }

    TEST(EndTime, NegativeUsesTheLatestLastTime)
    {
        EXPECT_DOUBLE_EQ(resolveEndTime(-1.0, loaders({0.3, 0.5, 0.1})), 0.5);
    }

    TEST(EndTime, NegativeIsAtLeastZero)
    {
        EXPECT_DOUBLE_EQ(resolveEndTime(-1.0, loaders({})), 0.0);
        EXPECT_DOUBLE_EQ(resolveEndTime(-1.0, loaders({-std::numeric_limits<double>::infinity()})), 0.0);
    }

    TEST(EndTime, NegativeIsUnboundedWithALiveSource)
    {
        EXPECT_TRUE(std::isinf(resolveEndTime(-1.0, loaders({0.5, std::nullopt}))));
    }
} // namespace Spacetime::Interface
