// Interface::ViewerSession over a mock session and a fake viewer: the order in which a session
// opens, starts, plays and runs a viewer, and how errors and early closes are handled.
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/interface/Viewer.hpp>
#include <spacetime/interface/ViewerSession.hpp>
#include "../TestEstimatorOptions.hpp"

namespace Spacetime::Interface
{
    namespace
    {
        using Log = std::vector<std::string>;

        /// A synchronous session that logs its hooks; start() or join() can be made to throw.
        struct MockSession
        {
            explicit MockSession(Log &log) : log(log) {}
            void start(Estimator &)
            {
                log.push_back("start");
                if (start_error)
                    throw std::invalid_argument("start");
            }
            void requestStop() { log.push_back("stop"); }
            void join()
            {
                log.push_back("join");
                if (join_error)
                    throw std::runtime_error("join");
            }
            bool finished() const { return true; }
            double startTime() const { return 1.0; }
            double endTime() const { return end_time; }
            double now() const { return 2.5; }
            bool isRealTime() const { return false; }

            Log &log;
            bool start_error = false, join_error = false;
            double end_time = 3.0;
        };

        using Session = ViewerSession<MockSession>;

        /// A viewer that logs its hooks; run() returns 7 unless closed first.
        struct FakeViewer : Viewer
        {
            FakeViewer(Log &log, bool online) : log(log), online(online) {}
            void open(Estimator &, ClosedCallback callback) override
            {
                log.push_back("open");
                if (open_error)
                    throw std::runtime_error("open");
                on_closed = std::move(callback);
            }
            bool isOnline() const override { return online; }
            void beginPlayback(double start_time, double end_time, LiveTimeSource live_time_source) override
            {
                log.push_back("play " + std::to_string(start_time) + " " + std::to_string(end_time) + (live_time_source ? " live" : ""));
                live = std::move(live_time_source);
            }
            int run() override
            {
                log.push_back(closed ? "run (closed)" : "run");
                if (run_error)
                    throw std::runtime_error("run");
                return closed ? 0 : 7;
            }
            void requestClose() override
            {
                log.push_back("close");
                closed = true;
            }

            Log &log;
            bool online;
            bool open_error = false, run_error = false, closed = false;
            ClosedCallback on_closed;
            LiveTimeSource live;
        };

        std::string play(double start_time, double end_time, bool live = false)
        {
            return "play " + std::to_string(start_time) + " " + std::to_string(end_time) + (live ? " live" : "");
        }

        class ViewerSessionTest : public ::testing::Test
        {
        protected:
            Log log;
            Estimator estimator{Testing::estimatorOptions(0.35)};
        };
    } // namespace

    TEST_F(ViewerSessionTest, WithoutAViewerItIsTheSession)
    {
        Session session(log);
        session.start(estimator);
        session.joinEstimation();
        session.join();
        EXPECT_EQ(log, (Log{"start", "join", "join"}));
        EXPECT_EQ(session.viewerExitCode(), 0);
    }

    // Offline: opened before estimation starts, replayed over [startTime, endTime] once it has ended.
    TEST_F(ViewerSessionTest, OfflineViewerOpensFirstAndReplaysAfterEstimation)
    {
        FakeViewer viewer(log, false);
        Session session(log);
        session.start(estimator, viewer);
        EXPECT_EQ(log, (Log{"open", "start"}));
        session.join();
        EXPECT_EQ(log, (Log{"open", "start", "join", play(1.0, 3.0), "run"}));
        EXPECT_EQ(session.viewerExitCode(), 7);
    }

    // Online: playback follows now() from the start, and the viewer runs while estimating.
    TEST_F(ViewerSessionTest, OnlineViewerPlaysFromTheStartAndRunsWhileEstimating)
    {
        FakeViewer viewer(log, true);
        Session session(log);
        session.start(estimator, viewer);
        EXPECT_EQ(log, (Log{"open", "start", play(1.0, 3.0, true)}));
        ASSERT_TRUE(viewer.live);
        EXPECT_EQ(viewer.live(), 2.5);
        session.join();
        EXPECT_EQ(log, (Log{"open", "start", play(1.0, 3.0, true), "run", "join"}));
    }

    TEST_F(ViewerSessionTest, ClosingTheViewerStopsTheSession)
    {
        FakeViewer viewer(log, true);
        Session session(log);
        session.start(estimator, viewer);
        ASSERT_TRUE(viewer.on_closed);
        viewer.on_closed();
        EXPECT_EQ(log.back(), "stop");
        session.join();
    }

    // Reporting results between estimation and an offline replay.
    TEST_F(ViewerSessionTest, JoinEstimationThenJoinRunsTheViewerOnly)
    {
        FakeViewer viewer(log, false);
        Session session(log);
        session.start(estimator, viewer);
        session.joinEstimation();
        EXPECT_EQ(log, (Log{"open", "start", "join"}));
        session.joinEstimation(); // already joined
        session.join();
        EXPECT_EQ(log, (Log{"open", "start", "join", play(1.0, 3.0), "run"}));
    }

    // An unbounded run (e.g. live data) replays up to where it stopped.
    TEST_F(ViewerSessionTest, UnboundedRunReplaysUpToNow)
    {
        FakeViewer viewer(log, false);
        Session session(log);
        session.end_time = std::numeric_limits<double>::infinity();
        session.start(estimator, viewer);
        session.join();
        EXPECT_EQ(log[3], play(1.0, 2.5));
    }

    TEST_F(ViewerSessionTest, ViewerThatFailsToOpenDoesNotStartTheSession)
    {
        FakeViewer viewer(log, false);
        viewer.open_error = true;
        Session session(log);
        EXPECT_THROW(session.start(estimator, viewer), std::runtime_error);
        EXPECT_EQ(log, (Log{"open"}));
    }

    TEST_F(ViewerSessionTest, FailedStartClosesTheViewerWithoutShowingIt)
    {
        FakeViewer viewer(log, false);
        Session session(log);
        session.start_error = true;
        EXPECT_THROW(session.start(estimator, viewer), std::invalid_argument);
        EXPECT_EQ(log, (Log{"open", "start", "close", "run (closed)"}));
    }

    TEST_F(ViewerSessionTest, EstimationErrorClosesTheViewerAndIsRethrown)
    {
        FakeViewer viewer(log, false);
        Session session(log);
        session.start(estimator, viewer);
        session.join_error = true;
        EXPECT_THROW(session.join(), std::runtime_error);
        EXPECT_EQ(log, (Log{"open", "start", "join", "close", "run (closed)"}));
    }

    TEST_F(ViewerSessionTest, ViewerErrorStopsEstimationAndIsRethrown)
    {
        FakeViewer viewer(log, true);
        viewer.run_error = true;
        Session session(log);
        session.start(estimator, viewer);
        EXPECT_THROW(session.join(), std::runtime_error);
        EXPECT_EQ(log, (Log{"open", "start", play(1.0, 3.0, true), "run", "stop", "join"}));
    }

    TEST_F(ViewerSessionTest, SessionDroppedBeforeJoinClosesTheViewer)
    {
        FakeViewer viewer(log, false);
        {
            Session session(log);
            session.start(estimator, viewer);
        }
        EXPECT_EQ(log, (Log{"open", "start", "close", "run (closed)"}));
    }

    TEST_F(ViewerSessionTest, ViewerDroppedFirstIsForgotten)
    {
        Session session(log);
        {
            FakeViewer viewer(log, false);
            session.start(estimator, viewer);
        }
        session.join();
        EXPECT_EQ(log, (Log{"open", "start", "join"}));
    }

    TEST_F(ViewerSessionTest, OnlyOneViewerPerSession)
    {
        FakeViewer first(log, false), second(log, false);
        Session session(log);
        session.start(estimator, first);
        EXPECT_THROW(session.start(estimator, second), std::logic_error);
        session.join();
    }
} // namespace Spacetime::Interface
