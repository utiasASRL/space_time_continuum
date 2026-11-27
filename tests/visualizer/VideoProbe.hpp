#ifndef TESTS_VIDEOPROBE_HPP
#define TESTS_VIDEOPROBE_HPP

#include <filesystem>
#include <optional>
#include <sstream>
#include <string>

#include <QProcess>
#include <QStandardPaths>
#include <QStringList>

namespace Spacetime::Tests
{
    /// Whether `program` (e.g. "ffmpeg") can be run from PATH.
    inline bool onPath(const std::string &program)
    {
        return !QStandardPaths::findExecutable(QString::fromStdString(program)).isEmpty();
    }

    /// What ffprobe reports for a video's first stream.
    struct VideoInfo
    {
        int frames = 0;
        int width = 0;
        int height = 0;
        double fps = 0.0;
    };

    /// Decodes `path` with ffprobe (counting every frame); nullopt if it cannot be read.
    inline std::optional<VideoInfo> probeVideo(const std::filesystem::path &path)
    {
        QProcess ffprobe;
        ffprobe.start("ffprobe", QStringList{"-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                                             "stream=nb_read_frames,width,height,r_frame_rate", "-of", "default=noprint_wrappers=1",
                                             QString::fromStdString(path.string())});
        if (!ffprobe.waitForFinished(-1) || ffprobe.exitStatus() != QProcess::NormalExit || ffprobe.exitCode() != 0)
            return std::nullopt;
        const std::string output = ffprobe.readAllStandardOutput().toStdString();

        VideoInfo info;
        std::istringstream lines(output);
        for (std::string line; std::getline(lines, line);)
        {
            const auto equals = line.find('=');
            if (equals == std::string::npos)
                continue;
            const std::string key = line.substr(0, equals), value = line.substr(equals + 1);
            if (key == "nb_read_frames")
                info.frames = std::stoi(value);
            else if (key == "width")
                info.width = std::stoi(value);
            else if (key == "height")
                info.height = std::stoi(value);
            else if (key == "r_frame_rate") // "num/den"
            {
                const auto slash = value.find('/');
                info.fps = slash == std::string::npos ? std::stod(value) : std::stod(value.substr(0, slash)) / std::stod(value.substr(slash + 1));
            }
        }
        return info;
    }
} // namespace Spacetime::Tests

#endif // TESTS_VIDEOPROBE_HPP
