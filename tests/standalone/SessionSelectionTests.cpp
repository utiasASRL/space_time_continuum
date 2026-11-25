// Selecting an interface through Spacetime::Session, with nothing but the package-level <spacetime> header.
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <spacetime>
#include "../TestEstimatorOptions.hpp"

namespace
{
    // No template argument: the default (standalone) interface, deduced from the constructor arguments too.
    static_assert(std::is_same_v<decltype(Spacetime::Session(std::declval<const Spacetime::SessionOptions &>())),
                                 Spacetime::Session<Spacetime::INTERFACE_STANDALONE>>);
    static_assert(std::is_same_v<decltype(Spacetime::Session()), Spacetime::Session<Spacetime::INTERFACE_STANDALONE>>);
    static_assert(std::is_same_v<Spacetime::Session<>, Spacetime::Session<Spacetime::INTERFACE_STANDALONE>>);
    static_assert(std::is_base_of_v<Spacetime::Interface::Standalone::Session, Spacetime::Session<>>);
    static_assert(std::is_same_v<Spacetime::SessionOptions, Spacetime::Interface::SessionOptions>);
    static_assert(Spacetime::Session<>::name == "default");
    static_assert(Spacetime::Interface::isBuilt<Spacetime::INTERFACE_STANDALONE>);
    static_assert(!Spacetime::Interface::isBuilt<Spacetime::INTERFACE_MUJOCO>);

    TEST(SessionSelection, DefaultSessionRunsTheEstimator)
    {
        Spacetime::Estimator estimator(Spacetime::Testing::estimatorOptions(0.0));
        Spacetime::SessionOptions options;
        options.start_time = 1.0;
        options.end_time = 1.2;
        options.period = 0.1;

        Spacetime::Session session(options);
        std::vector<double> ticks;
        session.setTickCallback([&ticks](double t)
                                { ticks.push_back(t); });
        session.start(estimator);
        session.join();

        EXPECT_TRUE(session.finished());
        EXPECT_FALSE(session.isRealTime());
        ASSERT_EQ(ticks.size(), 3u);
        EXPECT_DOUBLE_EQ(ticks.front(), 1.0);
        EXPECT_DOUBLE_EQ(session.endTime(), 1.2);
    }
} // namespace
