#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/interface/InterfaceConcept.hpp>
#include "../TestEstimatorOptions.hpp"

namespace Spacetime::Interface
{
    namespace
    {
        /// A minimal, synchronous Session: start() "runs" by recording the options' times.
        struct MockSession
        {
            explicit MockSession(const SessionOptions &options) : options(options) {}
            void setTickCallback(EstimatorDriver::TickCallback callback) { tick = std::move(callback); }
            void start(Estimator &)
            {
                started = true;
                if (tick)
                    tick(options.start_time);
            }
            void requestStop() { stop_requested = true; }
            void join() { joined = true; }
            bool finished() const { return joined; }
            double startTime() const { return options.start_time; }
            double endTime() const { return options.end_time; }
            double now() const { return options.start_time; }
            bool isRealTime() const { return options.real_time; }

            SessionOptions options;
            EstimatorDriver::TickCallback tick;
            bool started = false, stop_requested = false, joined = false;
        };

        struct MockInterface
        {
            static constexpr std::string_view name = "mock";
            using Session = MockSession;
        };

        // ---- Types that must not satisfy the concept ----

        struct NoSession
        {
            static constexpr std::string_view name = "no session";
        };

        struct NoName
        {
            using Session = MockSession;
        };

        struct WrongConstructorSession : MockSession
        {
            explicit WrongConstructorSession(Estimator &) : MockSession(SessionOptions()) {}
        };
        struct WrongConstructor
        {
            static constexpr std::string_view name = "wrong constructor";
            using Session = WrongConstructorSession;
        };

        /// MockSession minus one hook, by hiding it behind a private, inaccessible override.
        struct NoNowSession : MockSession
        {
            using MockSession::MockSession;

        private:
            double now() const;
        };
        struct NoNow
        {
            static constexpr std::string_view name = "no now";
            using Session = NoNowSession;
        };

        struct NoStopSession : MockSession
        {
            using MockSession::MockSession;

        private:
            void requestStop();
        };
        struct NoStop
        {
            static constexpr std::string_view name = "no stop";
            using Session = NoStopSession;
        };

        struct WrongReturnSession : MockSession
        {
            using MockSession::MockSession;
            bool start(Estimator &) { return true; } // must return void
        };
        struct WrongReturn
        {
            static constexpr std::string_view name = "wrong return";
            using Session = WrongReturnSession;
        };

        /// Generic downstream code: only the concept is known here.
        template <InterfaceType I>
        std::string_view runGeneric(Estimator &estimator, std::vector<double> &ticks)
        {
            SessionOptions options;
            options.start_time = 1.5;
            options.end_time = 2.0;
            typename I::Session session(options);
            session.setTickCallback([&ticks](double t)
                                    { ticks.push_back(t); });
            session.start(estimator);
            session.join();
            EXPECT_TRUE(session.finished());
            EXPECT_DOUBLE_EQ(session.endTime(), 2.0);
            EXPECT_FALSE(session.isRealTime());
            return I::name;
        }
    } // namespace

    TEST(InterfaceConcept, IncompleteInterfacesDoNotSatisfyIt)
    {
        static_assert(!InterfaceType<NoSession>);
        static_assert(!InterfaceType<NoName>);
        static_assert(!InterfaceType<WrongConstructor>);
        static_assert(!InterfaceType<NoNow>);
        static_assert(!InterfaceType<NoStop>);
        static_assert(!InterfaceType<WrongReturn>);
        static_assert(!InterfaceType<int>);
    }

    TEST(InterfaceConcept, GenericCodeRunsAgainstTheConceptAlone)
    {
        Estimator estimator(Testing::estimatorOptions(0.0));
        std::vector<double> ticks;
        EXPECT_EQ(runGeneric<MockInterface>(estimator, ticks), "mock");
        EXPECT_EQ(ticks, (std::vector<double>{1.5}));
    }
} // namespace Spacetime::Interface
