#include <gtest/gtest.h>

#include <spacetime/visualizer/PlaybackController.hpp>

namespace Spacetime
{
    using Spacetime::Visualization::PlaybackController;
    using Spacetime::Visualization::PlaybackState;

    TEST(PlaybackController, ImportTest)
    {
        EXPECT_TRUE(true);
    }

    TEST(PlaybackController, StartsPausedAtZero)
    {
        PlaybackController controller(/*online=*/false);
        EXPECT_EQ(controller.state(), PlaybackState::Paused);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 0.0);
    }

    TEST(PlaybackController, OfflineTickAdvancesTimeByWallDtWhilePlaying)
    {
        PlaybackController controller(/*online=*/false);
        controller.play();
        controller.tick(0.1);
        controller.tick(0.2);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 0.3);
    }

    TEST(PlaybackController, OfflinePauseFreezesDisplayedTime)
    {
        PlaybackController controller(/*online=*/false);
        controller.play();
        controller.tick(0.5);
        controller.pause();
        controller.tick(1.0); // must be ignored while paused
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 0.5);
    }

    TEST(PlaybackController, OfflineSpeedMultiplierScalesElapsedTime)
    {
        PlaybackController controller(/*online=*/false);
        controller.setSpeed(2.0);
        controller.play();
        controller.tick(1.0);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 2.0);
    }

    TEST(PlaybackController, OfflineSeekJumpsDirectly)
    {
        PlaybackController controller(/*online=*/false);
        controller.seek(5.0);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 5.0);
    }

    TEST(PlaybackController, OnlinePauseFreezesDisplayedTimeRegardlessOfLiveTime)
    {
        double fake_now = 10.0;
        PlaybackController controller(/*online=*/true);
        controller.setLiveTimeSource([&]()
                                     { return fake_now; });
        controller.play();
        controller.tick(0.1);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 10.0);

        controller.pause();
        fake_now = 50.0; // wall clock keeps moving while paused
        controller.tick(0.1);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 10.0);
    }

    // This is the core online-mode behavior the feature request calls for
    // explicitly: "When a user plays, jump to the current time" -- not resume
    // from wherever it was paused.
    TEST(PlaybackController, OnlinePlayAfterPauseJumpsToCurrentLiveTimeNotThePausedValue)
    {
        double fake_now = 10.0;
        PlaybackController controller(/*online=*/true);
        controller.setLiveTimeSource([&]()
                                     { return fake_now; });
        controller.play();
        controller.tick(0.1); // displayedTime() == 10.0
        controller.pause();

        fake_now = 47.0; // time passes elsewhere while paused
        controller.play();
        double displayed = controller.tick(0.1);

        EXPECT_DOUBLE_EQ(displayed, 47.0);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 47.0);
    }

    TEST(PlaybackController, OfflineSeekClampsToMinTime)
    {
        PlaybackController controller(/*online=*/false);
        controller.setTimeBounds(0.0, 20.0);
        controller.seek(-5.0);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 0.0);
    }

    TEST(PlaybackController, OfflineSeekClampsToMaxTime)
    {
        PlaybackController controller(/*online=*/false);
        controller.setTimeBounds(0.0, 20.0);
        controller.seek(50.0);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 20.0);
    }

    TEST(PlaybackController, OfflineTickPastMaxTimeClampsAndPauses)
    {
        PlaybackController controller(/*online=*/false);
        controller.setTimeBounds(0.0, 1.0);
        controller.seek(0.9);
        controller.play();
        controller.tick(1.0); // would advance to 1.9, past max_time
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 1.0);
        EXPECT_EQ(controller.state(), PlaybackState::Paused);
    }

    TEST(PlaybackController, OfflinePlayFromEndRestartsAtBeginning)
    {
        PlaybackController controller(/*online=*/false);
        controller.setTimeBounds(2.0, 10.0);
        controller.seek(10.0);
        controller.play(); // sitting at max_time -> should jump back to min_time
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 2.0);
        EXPECT_EQ(controller.state(), PlaybackState::Playing);
    }

    TEST(PlaybackController, OnlineModeUnaffectedByTimeBounds)
    {
        double fake_now = 10.0;
        PlaybackController controller(/*online=*/true);
        controller.setLiveTimeSource([&]()
                                     { return fake_now; });
        controller.setTimeBounds(0.0, 1.0); // deliberately smaller than fake_now
        controller.play();
        controller.tick(0.1);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 10.0);
        EXPECT_EQ(controller.state(), PlaybackState::Playing); // never auto-paused
    }

    TEST(PlaybackController, OnlineSeekAndSpeedAreNoOps)
    {
        double fake_now = 10.0;
        PlaybackController controller(/*online=*/true);
        controller.setLiveTimeSource([&]()
                                     { return fake_now; });
        controller.setSpeed(5.0);
        EXPECT_DOUBLE_EQ(controller.speed(), 1.0); // unchanged: setSpeed is offline-only

        controller.seek(99.0);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 0.0); // unchanged: seek is offline-only
    }

    TEST(PlaybackController, DelayDefaultsToZeroAndQueryTimeMatchesDisplayedTime)
    {
        PlaybackController controller(/*online=*/false);
        controller.seek(3.0);
        EXPECT_DOUBLE_EQ(controller.delay(), 0.0);
        EXPECT_DOUBLE_EQ(controller.queryTime(), controller.displayedTime());
    }

    TEST(PlaybackController, NegativeDelayClampsToZero)
    {
        PlaybackController controller(/*online=*/false);
        controller.setDelay(-1.0);
        EXPECT_DOUBLE_EQ(controller.delay(), 0.0);
    }

    TEST(PlaybackController, DelayIsUnboundedAboveWithoutMaxDelay)
    {
        PlaybackController controller(/*online=*/false);
        controller.setDelay(1e6);
        EXPECT_DOUBLE_EQ(controller.delay(), 1e6);
    }

    TEST(PlaybackController, DelayClampsToMaxDelay)
    {
        PlaybackController controller(/*online=*/false);
        controller.setMaxDelay(0.5);
        controller.setDelay(2.0);
        EXPECT_DOUBLE_EQ(controller.delay(), 0.5);
        controller.setDelay(0.5); // exactly at the window edge is allowed
        EXPECT_DOUBLE_EQ(controller.delay(), 0.5);
    }

    TEST(PlaybackController, LoweringMaxDelayReclampsCurrentDelay)
    {
        PlaybackController controller(/*online=*/false);
        controller.setDelay(1.0);
        controller.setMaxDelay(0.25);
        EXPECT_DOUBLE_EQ(controller.delay(), 0.25);
    }

    TEST(PlaybackController, NegativeMaxDelayIsTreatedAsZero)
    {
        PlaybackController controller(/*online=*/false);
        controller.setMaxDelay(-3.0);
        EXPECT_DOUBLE_EQ(controller.maxDelay(), 0.0);
        controller.setDelay(1.0);
        EXPECT_DOUBLE_EQ(controller.delay(), 0.0);
    }

    TEST(PlaybackController, OfflineQueryTimeTracksDisplayedTimeMinusDelay)
    {
        PlaybackController controller(/*online=*/false);
        controller.setTimeBounds(0.0, 20.0);
        controller.setDelay(0.4);
        controller.seek(5.0);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 5.0); // delay never moves the playback time itself
        EXPECT_DOUBLE_EQ(controller.queryTime(), 4.6);

        controller.play();
        controller.tick(1.0);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 6.0);
        EXPECT_DOUBLE_EQ(controller.queryTime(), 5.6);
    }

    TEST(PlaybackController, OnlineQueryTimeIsLiveTimeMinusDelayAndAdjustableWhilePaused)
    {
        double fake_now = 10.0;
        PlaybackController controller(/*online=*/true);
        controller.setLiveTimeSource([&]()
                                     { return fake_now; });
        controller.setDelay(1.5); // unlike seek/setSpeed, delay works online
        controller.play();
        controller.tick(0.1);
        EXPECT_DOUBLE_EQ(controller.queryTime(), 8.5);

        controller.pause();
        controller.setDelay(0.5);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 10.0);
        EXPECT_DOUBLE_EQ(controller.queryTime(), 9.5);
    }

    // Regression: online mode without a live time source has nothing to track, so
    // displayed time never leaves its initial 0 -- this is why offline_demos --online
    // drew nothing until it bound one (via Visualizer::setLiveTimeSource).
    TEST(PlaybackController, OnlineWithoutLiveTimeSourceNeverAdvances)
    {
        PlaybackController controller(/*online=*/true);
        controller.setTimeBounds(10.0, 20.0);
        controller.seek(10.0); // ignored online
        controller.play();
        controller.tick(0.5);
        controller.tick(0.5);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 0.0);
    }

    // Visualizer calls play() in its constructor, before the caller binds a live time
    // source, so a source bound after play() must still drive the very next tick.
    TEST(PlaybackController, OnlineLiveTimeSourceBoundAfterPlayIsHonoured)
    {
        double fake_now = 10.0;
        PlaybackController controller(/*online=*/true);
        controller.play();
        controller.setLiveTimeSource([&]()
                                     { return fake_now; });
        EXPECT_DOUBLE_EQ(controller.tick(0.033), 10.0);

        fake_now = 10.5;
        controller.setDelay(0.1); // phase delay still applies on top of the live time
        EXPECT_DOUBLE_EQ(controller.tick(0.033), 10.5);
        EXPECT_DOUBLE_EQ(controller.queryTime(), 10.4);
    }

    TEST(PlaybackController, DelayDoesNotAffectEndOfRangeAutoPause)
    {
        PlaybackController controller(/*online=*/false);
        controller.setTimeBounds(0.0, 1.0);
        controller.setDelay(0.5);
        controller.seek(0.9);
        controller.play();
        controller.tick(0.2); // displayedTime crosses max_time even though queryTime (0.6) has not
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 1.0);
        EXPECT_DOUBLE_EQ(controller.queryTime(), 0.5);
        EXPECT_EQ(controller.state(), PlaybackState::Paused);
    }

    TEST(PlaybackController, LoopingStartsOff)
    {
        EXPECT_FALSE(PlaybackController(/*online=*/false).looping());
    }

    TEST(PlaybackController, OfflineLoopingShowsTheEndThenRestartsAtTheBeginning)
    {
        PlaybackController controller(/*online=*/false);
        controller.setTimeBounds(2.0, 3.0);
        controller.setLooping(true);
        EXPECT_TRUE(controller.looping());
        controller.seek(2.9);
        controller.play();

        EXPECT_DOUBLE_EQ(controller.tick(1.0), 3.0) << "the last frame is still shown";
        EXPECT_EQ(controller.state(), PlaybackState::Playing) << "and playback carries on";
        EXPECT_DOUBLE_EQ(controller.tick(0.1), 2.0) << "from the beginning";
        EXPECT_EQ(controller.state(), PlaybackState::Playing);
        EXPECT_DOUBLE_EQ(controller.tick(0.1), 2.1);
    }

    TEST(PlaybackController, TurningLoopingOffPausesAtTheEndAgain)
    {
        PlaybackController controller(/*online=*/false);
        controller.setTimeBounds(0.0, 1.0);
        controller.setLooping(true);
        controller.setLooping(false);
        controller.seek(0.9);
        controller.play();
        controller.tick(1.0);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 1.0);
        EXPECT_EQ(controller.state(), PlaybackState::Paused);
    }

    TEST(PlaybackController, LoopingDoesNotMoveAPausedReplay)
    {
        PlaybackController controller(/*online=*/false);
        controller.setTimeBounds(0.0, 1.0);
        controller.setLooping(true);
        controller.seek(1.0);
        EXPECT_DOUBLE_EQ(controller.tick(0.5), 1.0);
        EXPECT_EQ(controller.state(), PlaybackState::Paused);
    }

    TEST(PlaybackController, OnlineIgnoresLooping)
    {
        double now = 5.0;
        PlaybackController controller(/*online=*/true);
        controller.setLiveTimeSource([&]() { return now; });
        controller.setTimeBounds(0.0, 1.0);
        controller.setLooping(true);
        EXPECT_FALSE(controller.looping()) << "a live stream has no end to loop from";
        controller.play();
        EXPECT_DOUBLE_EQ(controller.tick(0.1), 5.0);
        now = 6.0;
        EXPECT_DOUBLE_EQ(controller.tick(0.1), 6.0);
    }

    TEST(PlaybackController, StepTickMovesOneTickFromATick)
    {
        PlaybackController controller(/*online=*/false);
        controller.setTimeBounds(10.0, 20.0); // ticks at 10 + k / 30
        const double period = 1.0 / 30.0;
        controller.seek(10.0 + 3 * period);
        controller.stepTick(period, +1);
        EXPECT_NEAR(controller.displayedTime(), 10.0 + 4 * period, 1e-12);
        controller.stepTick(period, -1);
        controller.stepTick(period, -1);
        EXPECT_NEAR(controller.displayedTime(), 10.0 + 2 * period, 1e-12);
    }

    TEST(PlaybackController, StepTickSnapsToTheNextTickInTheStepDirection)
    {
        PlaybackController controller(/*online=*/false);
        controller.setTimeBounds(0.0, 1.0);
        controller.seek(0.23);
        controller.stepTick(0.1, +1);
        EXPECT_NEAR(controller.displayedTime(), 0.3, 1e-12);
        controller.seek(0.27);
        controller.stepTick(0.1, -1);
        EXPECT_NEAR(controller.displayedTime(), 0.2, 1e-12);
    }

    TEST(PlaybackController, StepTickTreatsRoundingErrorsAsOnTheTick)
    {
        // 0.1 * 3 is 0.30000000000000004: still the 3rd tick, so stepping moves a whole tick.
        PlaybackController controller(/*online=*/false);
        controller.setTimeBounds(0.0, 1.0);
        controller.seek(0.1 * 3);
        controller.stepTick(0.1, +1);
        EXPECT_NEAR(controller.displayedTime(), 0.4, 1e-12);
        controller.seek(0.1 * 3);
        controller.stepTick(0.1, -1);
        EXPECT_NEAR(controller.displayedTime(), 0.2, 1e-12);
    }

    TEST(PlaybackController, StepTickClampsToTheTimeBounds)
    {
        PlaybackController controller(/*online=*/false);
        controller.setTimeBounds(0.0, 0.95); // the end is not a tick
        controller.seek(0.0);
        controller.stepTick(0.1, -1);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 0.0);
        controller.seek(0.9);
        controller.stepTick(0.1, +1);
        EXPECT_DOUBLE_EQ(controller.displayedTime(), 0.95);
        controller.stepTick(0.1, -1); // back from the end: the last tick
        EXPECT_NEAR(controller.displayedTime(), 0.9, 1e-12);
    }

    TEST(PlaybackController, StepTickIsANoOpOnlineOrWithoutAPeriod)
    {
        PlaybackController online(/*online=*/true);
        online.stepTick(0.1, +1);
        EXPECT_DOUBLE_EQ(online.displayedTime(), 0.0);

        PlaybackController offline(/*online=*/false);
        offline.setTimeBounds(0.0, 1.0);
        offline.seek(0.5);
        offline.stepTick(0.0, +1);
        offline.stepTick(-0.1, +1);
        offline.stepTick(0.1, 0);
        EXPECT_DOUBLE_EQ(offline.displayedTime(), 0.5);
    }
}
