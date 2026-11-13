#ifndef SPACETIME_INTERFACE_ENDTIME_HPP
#define SPACETIME_INTERFACE_ENDTIME_HPP

#include <algorithm>
#include <limits>

namespace Spacetime::Interface
{
    /**
     * @brief The end time to estimate until. A non-negative `end_time` is kept; a negative
     *        one means "until the data ends": the latest Dataloader::lastTime() among
     *        `dataloaders` (at least 0), or unbounded (+infinity, until stopped) if any of
     *        them is live.
     * @param dataloaders Any range of (smart) pointers to Interface::Dataloader or derived types.
     */
    template <typename Dataloaders>
    double resolveEndTime(double end_time, const Dataloaders &dataloaders)
    {
        if (end_time >= 0)
            return end_time;
        double data_end_time = 0.0;
        for (const auto &dataloader : dataloaders)
        {
            const auto last_time = dataloader->lastTime();
            if (!last_time)
                return std::numeric_limits<double>::infinity();
            data_end_time = std::max(data_end_time, *last_time);
        }
        return data_end_time;
    }
} // namespace Spacetime::Interface

#endif // SPACETIME_INTERFACE_ENDTIME_HPP
