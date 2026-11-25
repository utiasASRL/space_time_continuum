#include <sys/resource.h>

#include "ThreadPriority.hpp"

namespace Spacetime::Interface::Standalone
{
    void lowerCurrentThreadPriority(int nice_delta)
    {
        setpriority(PRIO_PROCESS, 0, getpriority(PRIO_PROCESS, 0) + nice_delta);
    }
} // namespace Spacetime::Interface::Standalone
