#include <filesystem>
#include <stdexcept>

#include <QColor>
#include <QImage>
#include <QSize>
#include <gtest/gtest.h>

#include <spacetime/visualizer/default/VideoWriter.hpp>

#include "VideoProbe.hpp"

namespace Spacetime::Visualization::Default
{
    namespace
    {
        /// A fresh scratch folder per test, removed afterwards.
        class VideoWriterTest : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                if (!Tests::onPath("ffmpeg") || !Tests::onPath("ffprobe"))
                    GTEST_SKIP() << "ffmpeg/ffprobe not on PATH";
                folder = std::filesystem::temp_directory_path() / ("spacetime_video_writer_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
                std::filesystem::remove_all(folder);
            }
            void TearDown() override { std::filesystem::remove_all(folder); }

            static QImage solid(QSize size, Qt::GlobalColor color)
            {
                QImage image(size, QImage::Format_ARGB32);
                image.fill(color);
                return image;
            }

            std::filesystem::path folder;
        };
    } // namespace

    TEST_F(VideoWriterTest, WritesEveryFrameAtTheGivenRate)
    {
        const auto path = folder / "nested" / "video.mp4"; // the parent folders are created
        VideoWriter writer(path, QSize(64, 48), 10.0);
        for (const auto color : {Qt::red, Qt::green, Qt::blue, Qt::white, Qt::black})
            writer.write(solid(QSize(64, 48), color));
        EXPECT_EQ(writer.frames(), 5u);
        writer.finish();

        const auto info = Tests::probeVideo(path);
        ASSERT_TRUE(info.has_value());
        EXPECT_EQ(info->frames, 5);
        EXPECT_EQ(info->width, 64);
        EXPECT_EQ(info->height, 48);
        EXPECT_DOUBLE_EQ(info->fps, 10.0);
    }

    TEST_F(VideoWriterTest, OddSizesAreRoundedDownToEven)
    {
        const auto path = folder / "odd.mp4";
        VideoWriter writer(path, QSize(65, 47), 20.0);
        EXPECT_EQ(writer.size(), QSize(64, 46));
        writer.write(solid(QSize(65, 47), Qt::red));
        writer.finish();

        const auto info = Tests::probeVideo(path);
        ASSERT_TRUE(info.has_value());
        EXPECT_EQ(info->width, 64);
        EXPECT_EQ(info->height, 46);
    }

    TEST_F(VideoWriterTest, FramesOfAnotherSizeAreScaledToTheVideo)
    {
        // E.g. the window was resized mid-export.
        const auto path = folder / "resized.mp4";
        VideoWriter writer(path, QSize(64, 48), 10.0);
        writer.write(solid(QSize(64, 48), Qt::red));
        writer.write(solid(QSize(128, 30), Qt::blue));
        writer.finish();

        const auto info = Tests::probeVideo(path);
        ASSERT_TRUE(info.has_value());
        EXPECT_EQ(info->frames, 2);
        EXPECT_EQ(info->width, 64);
        EXPECT_EQ(info->height, 48);
    }

    TEST_F(VideoWriterTest, FinishingTwiceIsANoOpAndTheDestructorFinishes)
    {
        const auto finished = folder / "finished.mp4";
        {
            VideoWriter writer(finished, QSize(32, 32), 10.0);
            writer.write(solid(QSize(32, 32), Qt::red));
            writer.finish();
            EXPECT_NO_THROW(writer.finish());
        }
        const auto destroyed = folder / "destroyed.mp4";
        {
            VideoWriter writer(destroyed, QSize(32, 32), 10.0);
            writer.write(solid(QSize(32, 32), Qt::red));
        }
        ASSERT_TRUE(Tests::probeVideo(finished).has_value());
        const auto info = Tests::probeVideo(destroyed);
        ASSERT_TRUE(info.has_value());
        EXPECT_EQ(info->frames, 1);
    }

    TEST_F(VideoWriterTest, WritingAfterFinishThrows)
    {
        VideoWriter writer(folder / "closed.mp4", QSize(32, 32), 10.0);
        writer.finish();
        EXPECT_THROW(writer.write(solid(QSize(32, 32), Qt::red)), std::logic_error);
    }

    TEST_F(VideoWriterTest, InvalidArgumentsAreRejected)
    {
        EXPECT_THROW(VideoWriter(folder / "a.mp4", QSize(1, 1), 10.0), std::invalid_argument); // rounds to 0 x 0
        EXPECT_THROW(VideoWriter(folder / "b.mp4", QSize(32, 32), 0.0), std::invalid_argument);
        EXPECT_THROW(VideoWriter(folder / "c.mp4", QSize(32, 32), -5.0), std::invalid_argument);
    }

    TEST_F(VideoWriterTest, AMissingEncoderThrows)
    {
        EXPECT_THROW(VideoWriter(folder / "missing.mp4", QSize(32, 32), 10.0, "spacetime-no-such-ffmpeg"), std::runtime_error);
    }
} // namespace Spacetime::Visualization::Default
