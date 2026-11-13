#include <limits>

#include <gtest/gtest.h>

#include <spacetime/interface/Dataloader.hpp>

namespace Spacetime::Interface
{
    namespace
    {
        /// Exposes the protected watermark setters, and counts the watermark-moved hook.
        struct ManualDataloader : public Dataloader
        {
            using Dataloader::finish;
            using Dataloader::setNextTime;
            int moves = 0;

        protected:
            void onWatermarkMoved() override { moves++; }
        };
    } // namespace

    TEST(Dataloader, NothingIsDeliveredInitially)
    {
        ManualDataloader loader;
        EXPECT_FALSE(loader.deliveredThrough(-1e9));
        EXPECT_FALSE(loader.deliveredThrough(0.0));
    }

    TEST(Dataloader, HasNoKnownLastTimeByDefault)
    {
        ManualDataloader loader;
        EXPECT_FALSE(loader.lastTime().has_value());
    }

    // setNextTime(t) means: everything stamped before t has been delivered, and the next
    // measurement may be stamped exactly t, so t itself is not yet complete.
    TEST(Dataloader, NextTimeIsExclusive)
    {
        ManualDataloader loader;
        loader.setNextTime(0.5);
        EXPECT_TRUE(loader.deliveredThrough(0.0));
        EXPECT_TRUE(loader.deliveredThrough(0.4999));
        EXPECT_FALSE(loader.deliveredThrough(0.5));
        EXPECT_FALSE(loader.deliveredThrough(1.0));
    }

    TEST(Dataloader, FinishDeliversEverything)
    {
        ManualDataloader loader;
        loader.setNextTime(0.5);
        loader.finish();
        EXPECT_TRUE(loader.deliveredThrough(0.5));
        EXPECT_TRUE(loader.deliveredThrough(std::numeric_limits<double>::max()));
    }

    TEST(Dataloader, WatermarkNeverMovesBackwards)
    {
        ManualDataloader loader;
        loader.setNextTime(1.0);
        loader.setNextTime(0.5);
        EXPECT_TRUE(loader.deliveredThrough(0.9));
    }

    // The hook (e.g. a threaded interface waking its waiters) fires only on actual progress.
    TEST(Dataloader, HookFiresOnlyWhenTheWatermarkMoves)
    {
        ManualDataloader loader;
        loader.setNextTime(1.0);
        EXPECT_EQ(loader.moves, 1);
        loader.setNextTime(0.5); // backwards: ignored
        loader.setNextTime(1.0); // no progress
        EXPECT_EQ(loader.moves, 1);
        loader.finish();
        EXPECT_EQ(loader.moves, 2);
        loader.finish(); // already finished
        EXPECT_EQ(loader.moves, 2);
    }
} // namespace Spacetime::Interface
