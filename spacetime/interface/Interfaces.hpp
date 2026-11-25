/**
 * @file Interfaces.hpp
 * @brief Every interface spacetime was built with (see the Interface* CMake options), and
 *        Interface::Session, the session of the one selected by its template argument (one of
 *        the INTERFACE_* constants, see InterfaceConcept.hpp). Downstream code links
 *        spacetime::interfaces and includes <spacetime> (or <spacetime/Types.hpp>), which brings this in and
 *        names it Spacetime::Session:
 *        @code
 *        Spacetime::Session session(options);                                    // the default (standalone) interface
 *        Spacetime::Session<Spacetime::INTERFACE_ROS2> ros(options, ros_options); // the ROS2 interface
 *        session.start(estimator);
 *        @endcode
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_INTERFACES_HPP
#define SPACETIME_INTERFACES_HPP

#include <string_view>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/interface/InterfaceConcept.hpp>

#if defined(SPACETIME_INTERFACE_STANDALONE) && SPACETIME_INTERFACE_STANDALONE
#include <spacetime/interface/standalone/Interface.hpp>
#endif

#if defined(SPACETIME_INTERFACE_ROS2) && SPACETIME_INTERFACE_ROS2
#include <spacetime/interface/ros2/Interface.hpp>
#endif

namespace Spacetime::Interface
{
    /// Whether spacetime was built with interface `I` (one of the INTERFACE_* constants), i.e. whether Traits<I> is defined.
    template <int I>
    inline constexpr bool isBuilt = requires { sizeof(Traits<I>); };

    namespace Detail
    {
        /// Traits<I>::Session, or an empty placeholder for an interface that was not built (so Session's static_asserts report it).
        template <int I, bool = isBuilt<I>>
        struct SessionBase
        {
            using Type = typename Traits<I>::Session;
            static constexpr std::string_view name = Traits<I>::name;
        };
        template <int I>
        struct SessionBase<I, false>
        {
            struct Type
            {
            };
            static constexpr std::string_view name = "";
        };
    } // namespace Detail

    /**
     * @brief The session of interface `I`: that interface's own Session (e.g. Ros2::Session, with
     *        its extra constructor arguments and methods), selected at compile time. See
     *        InterfaceConcept.hpp for the hooks every session provides.
     * @tparam I One of the INTERFACE_* constants; the standalone ("default") interface if omitted.
     */
    template <int I = INTERFACE_STANDALONE>
    class Session : public Detail::SessionBase<I>::Type
    {
        static_assert(I != INTERFACE_MUJOCO, "Spacetime::Session: the MuJoCo interface is not implemented yet.");
        static_assert(I == INTERFACE_STANDALONE || I == INTERFACE_ROS2 || I == INTERFACE_MUJOCO,
                      "Spacetime::Session: the interface must be one of the INTERFACE_* constants.");
        static_assert(I == INTERFACE_MUJOCO || (I != INTERFACE_STANDALONE && I != INTERFACE_ROS2) || isBuilt<I>, "Spacetime::Session: spacetime was built without this interface "
                                  "(see the InterfaceStandalone and InterfaceROS2 CMake options).");

    public:
        using Detail::SessionBase<I>::Type::Type;

        /// The interface's name, e.g. "default" or "ros2".
        static constexpr std::string_view name = Detail::SessionBase<I>::name;
    };

    /// `Spacetime::Session session;` selects the default interface.
    Session() -> Session<>;
    /// `Spacetime::Session session(options);` selects the default interface.
    Session(const SessionOptions &) -> Session<>;
} // namespace Spacetime::Interface

#endif // SPACETIME_INTERFACES_HPP
