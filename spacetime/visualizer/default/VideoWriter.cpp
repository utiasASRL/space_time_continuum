#include <spacetime/visualizer/default/VideoWriter.hpp>

#include <stdexcept>

#include <QStringList>

#include <spacetime/estimator/utilities/logger.hpp>

namespace Spacetime::Visualization::Default
{
    namespace
    {
        /// How long the encoder may take to start, accept a frame, or write the file [ms].
        constexpr int kStartTimeoutMs = 5000;
        constexpr int kWriteTimeoutMs = 30000;
        constexpr int kFinishTimeoutMs = 120000;
    } // namespace

    VideoWriter::VideoWriter(const std::filesystem::path &path, QSize size, double fps, const std::string &program)
        : m_path(path), m_size(size.width() & ~1, size.height() & ~1)
    {
        if (m_size.isEmpty())
            throw std::invalid_argument("VideoWriter: the frame size must be at least 2 x 2 pixels.");
        if (!(fps > 0.0))
            throw std::invalid_argument("VideoWriter: the frame rate must be positive.");
        if (m_path.has_parent_path())
            std::filesystem::create_directories(m_path.parent_path());

        const QStringList arguments{
            "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", QString("%1x%2").arg(m_size.width()).arg(m_size.height()),
            "-r", QString::number(fps, 'g', 17),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            QString::fromStdString(m_path.string())};
        m_process.setProcessChannelMode(QProcess::SeparateChannels);
        m_process.setStandardOutputFile(QProcess::nullDevice());
        m_process.start(QString::fromStdString(program), arguments);
        if (!m_process.waitForStarted(kStartTimeoutMs))
            throw std::runtime_error("VideoWriter: could not start \"" + program + "\" (is ffmpeg installed and on PATH?): " +
                                     m_process.errorString().toStdString());
    }

    VideoWriter::~VideoWriter()
    {
        try
        {
            finish();
        }
        catch (const std::exception &e)
        {
            error() << e.what();
        }
    }

    void VideoWriter::write(const QImage &frame)
    {
        if (m_finished)
            throw std::logic_error("VideoWriter::write called after finish().");
        QImage image = frame.size() == m_size ? frame : frame.scaled(m_size, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
        image.convertTo(QImage::Format_RGBA8888);

        // Rows may be padded (bytesPerLine > width * 4), so they are written one by one.
        const qint64 row_bytes = static_cast<qint64>(m_size.width()) * 4;
        for (int y = 0; y < image.height(); y++)
            if (m_process.write(reinterpret_cast<const char *>(image.constScanLine(y)), row_bytes) != row_bytes)
                throw std::runtime_error("VideoWriter: the encoder stopped accepting frames: " + m_process.errorString().toStdString());
        // Without an event loop the pipe only drains while waiting on it.
        while (m_process.bytesToWrite() > 0)
            if (!m_process.waitForBytesWritten(kWriteTimeoutMs))
                throw std::runtime_error("VideoWriter: the encoder stopped accepting frames: " + m_process.errorString().toStdString());
        m_frames++;
    }

    void VideoWriter::finish()
    {
        if (m_finished)
            return;
        m_finished = true;
        m_process.closeWriteChannel();
        if (!m_process.waitForFinished(kFinishTimeoutMs))
        {
            m_process.kill();
            m_process.waitForFinished();
            throw std::runtime_error("VideoWriter: the encoder did not finish writing " + m_path.string() + ".");
        }
        if (m_process.exitStatus() != QProcess::NormalExit || m_process.exitCode() != 0)
            throw std::runtime_error("VideoWriter: the encoder failed writing " + m_path.string() + ": " +
                                     m_process.readAllStandardError().trimmed().toStdString());
    }
} // namespace Spacetime::Visualization::Default
