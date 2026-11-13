#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <sstream>
#include <string>
#include <streambuf>
#include <memory>

// ANSI color codes
#define COLOR_RED "\033[31m"
#define COLOR_ORANGE "\033[38;5;208m" // Bright orange
#define COLOR_YELLOW "\033[33m"
#define COLOR_WHITE "\033[37m"
#define COLOR_RESET "\033[0m"

#ifndef SPACETIME_ENABLE_DEBUG_LOG
#define SPACETIME_ENABLE_DEBUG_LOG 0
#endif

/// Severity of a log message.
enum class LogLevel
{
    ERROR,
    WARNING,
    DEBUG,
    INFO
};

/// ANSI color code for `level` ("" for DEBUG when debug logging is compiled out).
inline std::string levelColor(LogLevel level)
{
#if !SPACETIME_ENABLE_DEBUG_LOG
    if (level == LogLevel::DEBUG)
    {
        return "";
    }
#endif

    switch (level)
    {
    case LogLevel::ERROR:
        return COLOR_RED;
    case LogLevel::WARNING:
        return COLOR_ORANGE;
    case LogLevel::DEBUG:
        return COLOR_YELLOW;
    case LogLevel::INFO:
        return COLOR_WHITE;
    }
    return COLOR_RESET;
}

/// Bracketed label for `level`, e.g. "[INFO]" ("" for DEBUG when debug logging is compiled out).
inline std::string levelLabel(LogLevel level)
{
#if !SPACETIME_ENABLE_DEBUG_LOG
    if (level == LogLevel::DEBUG)
    {
        return "";
    }
#endif

    switch (level)
    {
    case LogLevel::ERROR:
        return "[ERROR]";
    case LogLevel::WARNING:
        return "[WARNING]";
    case LogLevel::DEBUG:
        return "[DEBUG]";
    case LogLevel::INFO:
        return "[INFO]";
    }
    return "[LOG]";
}

// --- Base stream-style logger ---
/// Stream-style log line: buffers what is streamed into it and prints it to stdout on destruction.
class Logger
{
public:
    /// Print the level's color and label now (unless it is a disabled DEBUG logger).
    Logger(LogLevel level)
        : level_(level), enabled_(!(level == LogLevel::DEBUG && !SPACETIME_ENABLE_DEBUG_LOG))
    {
        if (enabled_)
        {
            std::cout << levelColor(level_) << levelLabel(level_) << " ";
        }
        buffer_.str("");
        buffer_.clear();
    }

    /// Print the buffered message and end the line.
    ~Logger()
    {
        if (enabled_)
        {
            std::cout << buffer_.str() << COLOR_RESET << std::endl;
        }
    }

    /// Append `value` to the message.
    template <typename T>
    Logger &operator<<(const T &value)
    {
        buffer_ << value;
        return *this;
    }

    /// Apply a stream manipulator (e.g. std::endl) to the message.
    Logger &operator<<(std::ostream &(*manip)(std::ostream &))
    {
        manip(buffer_);
        return *this;
    }

private:
    LogLevel level_;
    bool enabled_;
    std::ostringstream buffer_;
};

// --- Direct function-style log ---
/// Print `msg` at `level` as one line (DEBUG is dropped unless SPACETIME_ENABLE_DEBUG_LOG).
inline void logMessage(LogLevel level, const std::string &msg)
{
#if !SPACETIME_ENABLE_DEBUG_LOG
    if (level == LogLevel::DEBUG)
    {
        return;
    }
#endif

    std::cout << levelColor(level) << levelLabel(level)
              << " " << msg << COLOR_RESET << std::endl;
}

// --- Stream-style factories ---
/// An ERROR Logger; stream into it, e.g. `error() << "x = " << x;`.
inline Logger error() { return Logger(LogLevel::ERROR); }
/// A WARNING Logger.
inline Logger warning() { return Logger(LogLevel::WARNING); }
/// A DEBUG Logger (silent unless SPACETIME_ENABLE_DEBUG_LOG).
inline Logger debug() { return Logger(LogLevel::DEBUG); }
/// An INFO Logger.
inline Logger info() { return Logger(LogLevel::INFO); }

// --- Function-style helpers ---
/// logMessage at ERROR level.
inline void error(const std::string &msg) { logMessage(LogLevel::ERROR, msg); }
/// logMessage at WARNING level.
inline void warning(const std::string &msg) { logMessage(LogLevel::WARNING, msg); }
/// logMessage at DEBUG level.
inline void debug(const std::string &msg) { logMessage(LogLevel::DEBUG, msg); }
/// logMessage at INFO level.
inline void info(const std::string &msg) { logMessage(LogLevel::INFO, msg); }

// --- Redirected stream mode ---
/// Captures everything written to std::cout until logReset(), then prints it under one level label.
class RedirectLogger
{
public:
    /// Redirect std::cout into the buffer, starting it with the level's color and label.
    RedirectLogger(LogLevel level)
        : level_(level), old_buf_(std::cout.rdbuf(buffer_.rdbuf()))
    {
        std::cout << levelColor(level_) << levelLabel(level_) << std::endl;
    }

    /// Restore std::cout and print the captured output.
    void logReset()
    {
        std::cout.rdbuf(old_buf_);
        std::cout << buffer_.str() << COLOR_RESET << std::endl;
    }

private:
    LogLevel level_;
    std::ostringstream buffer_;
    std::streambuf *old_buf_;
};

// global redirector instance
inline std::unique_ptr<RedirectLogger> redirectInstance;

/// Start capturing std::cout at ERROR level (see RedirectLogger); end with logReset() before starting another.
inline void errorStart() { redirectInstance = std::make_unique<RedirectLogger>(LogLevel::ERROR); }
/// As errorStart, at WARNING level.
inline void warningStart() { redirectInstance = std::make_unique<RedirectLogger>(LogLevel::WARNING); }
/// As errorStart, at DEBUG level (no-op unless SPACETIME_ENABLE_DEBUG_LOG).
inline void debugStart()
{
#if SPACETIME_ENABLE_DEBUG_LOG
    redirectInstance = std::make_unique<RedirectLogger>(LogLevel::DEBUG);
#endif
}
/// As errorStart, at INFO level.
inline void infoStart() { redirectInstance = std::make_unique<RedirectLogger>(LogLevel::INFO); }

/// End the capture started by an *Start() call, printing it. No-op if none is active.
inline void logReset()
{
    if (redirectInstance)
    {
        redirectInstance->logReset();
        redirectInstance.reset();
    }
}
#endif // LOGGER_H