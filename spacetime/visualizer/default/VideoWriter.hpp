#ifndef DEFAULT_VIDEOWRITER_HPP
#define DEFAULT_VIDEOWRITER_HPP

#include <cstddef>
#include <filesystem>
#include <string>

#include <QImage>
#include <QProcess>
#include <QSize>

namespace Spacetime::Visualization::Default
{
    /**
     * @brief Encodes frames into an H.264 mp4 by streaming them, as raw RGBA, to an ffmpeg
     *        process (which must be on PATH: VTK's own movie writers are not built here).
     *        The video's size is fixed at construction; frames of another size (e.g. after
     *        a window resize) are scaled to it. Not copyable; finished on destruction.
     */
    class VideoWriter
    {
    public:
        /**
         * @param path    Output file; its parent folders are created and an existing file is overwritten.
         * @param size    Frame size; rounded down to even dimensions, as yuv420p requires.
         * @param fps     Playback rate of the video [frames/s].
         * @param program The encoder executable.
         * @throws std::invalid_argument if the rounded size is empty or `fps` is not positive.
         * @throws std::runtime_error if the encoder cannot be started.
         */
        VideoWriter(const std::filesystem::path &path, QSize size, double fps, const std::string &program = "ffmpeg");
        /// Finishes the video if finish() was not called, logging (not throwing) any error.
        ~VideoWriter();

        VideoWriter(const VideoWriter &) = delete;
        VideoWriter &operator=(const VideoWriter &) = delete;

        /**
         * @brief Appends `frame`, scaled to size() if needed.
         * @throws std::logic_error after finish().
         * @throws std::runtime_error if the encoder stopped accepting frames.
         */
        void write(const QImage &frame);

        /**
         * @brief Closes the stream and waits for the encoder to write the file. No-op if already finished.
         * @throws std::runtime_error if the encoder failed (with its error output).
         */
        void finish();

        /// The (even) frame size of the video.
        QSize size() const { return m_size; }
        /// Number of frames written so far.
        std::size_t frames() const { return m_frames; }
        /// The output file.
        const std::filesystem::path &path() const { return m_path; }

    private:
        std::filesystem::path m_path;
        QSize m_size;
        QProcess m_process;
        std::size_t m_frames = 0;
        bool m_finished = false;
    };
} // namespace Spacetime::Visualization::Default

#endif // DEFAULT_VIDEOWRITER_HPP
