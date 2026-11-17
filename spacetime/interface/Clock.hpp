/**
 * @file Clock.hpp
 * @brief The source of estimation time. An interface supplies the clock that matches where
 *        its data comes from, e.g. the standalone interface's offline and wall clocks
 *        (see spacetime/interface/standalone/Clock.hpp); a simulator or middleware
 *        interface would read its own time (e.g. Mujoco sim time, a ROS2 /clock). How,
 *        and on which thread, estimation is paced by it is up to the interface.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_INTERFACE_CLOCK_HPP
#define SPACETIME_INTERFACE_CLOCK_HPP

namespace Spacetime::Interface
{
    /// A source of estimation time, in seconds. now() must be safe to call from any thread.
    struct Clock
    {
        virtual ~Clock() = default;

        /// The current time, in seconds.
        virtual double now() const = 0;
        /// True if time passes on its own (so estimation must not wait on data); false if only the interface advances it.
        virtual bool isRealTime() const = 0;
    };
} // namespace Spacetime::Interface

#endif // SPACETIME_INTERFACE_CLOCK_HPP
