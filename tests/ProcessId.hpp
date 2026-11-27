#ifndef TESTS_PROCESSID_HPP
#define TESTS_PROCESSID_HPP

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace Spacetime::Testing
{
    /// This process's id, e.g. to name files or ROS entities apart from concurrently running tests.
    inline int processId()
    {
#if defined(_WIN32)
        return _getpid();
#else
        return static_cast<int>(::getpid());
#endif
    }
} // namespace Spacetime::Testing

#endif // TESTS_PROCESSID_HPP
