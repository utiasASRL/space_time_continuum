#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/interface/InterfaceConcept.hpp>

namespace Spacetime::Interface
{
    namespace
    {
        using Queues = std::vector<Types::pMeasurementQueueType>;

        /// A minimal, synchronous Session: start() "runs" by recording the options' times.
        struct MockSession
        {
            MockSession(Estimator &, const SessionOptions &options) : options(options) {}
            void addMeasurementQueues(const Queues &queues) { queue_count += queues.size(); }
            void setTickCallback(EstimatorDriver::TickCallback callback) { tick = std::move(callback); }
            void start()
            {
                started = true;
                if (tick)
                    tick(options.start_time);
            }
            void requestStop() { stop_requested = true; }
            void join() { joined = true; }
            bool finished() const { return joined; }
            double endTime() const { return options.end_time; }
            double now() const { return options.start_time; }
            bool isRealTime() const { return options.real_time; }

            SessionOptions options;
            EstimatorDriver::TickCallback tick;
            std::size_t queue_count = 0;
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
            explicit WrongConstructorSession(Estimator &estimator) : MockSession(estimator, SessionOptions()) {}
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

        struct NoQueuesSession : MockSession
        {
            using MockSession::MockSession;

        private:
            void addMeasurementQueues(const Queues &);
        };
        struct NoQueues
        {
            static constexpr std::string_view name = "no queues";
            using Session = NoQueuesSession;
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
            bool start() { return true; } // must return void
        };
        struct WrongReturn
        {
            static constexpr std::string_view name = "wrong return";
            using Session = WrongReturnSession;
        };

        /// Generic downstream code: only the concept is known here.
        template <InterfaceType I>
        std::string_view runGeneric(Estimator &estimator, const Queues &queues, std::vector<double> &ticks)
        {
            SessionOptions options;
            options.start_time = 1.5;
            options.end_time = 2.0;
            typename I::Session session(estimator, options);
            session.addMeasurementQueues(queues);
            session.setTickCallback([&ticks](double t)
                                    { ticks.push_back(t); });
            session.start();
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
        static_assert(!InterfaceType<NoQueues>);
        static_assert(!InterfaceType<NoStop>);
        static_assert(!InterfaceType<WrongReturn>);
        static_assert(!InterfaceType<int>);
    }

    TEST(InterfaceConcept, GenericCodeRunsAgainstTheConceptAlone)
    {
        Estimator estimator;
        std::vector<double> ticks;
        const Queues queues = {std::make_shared<Types::MeasurementQueue>()};
        EXPECT_EQ(runGeneric<MockInterface>(estimator, queues, ticks), "mock");
        EXPECT_EQ(ticks, (std::vector<double>{1.5}));
    }
} // namespace Spacetime::Interface
