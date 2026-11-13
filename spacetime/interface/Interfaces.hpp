/**
 * @file Interfaces.hpp
 * @brief Every interface spacetime was built with (see the Interface* CMake options), as
 *        Spacetime::Interfaces::<Name> bundles satisfying Interface::InterfaceType. Link
 *        spacetime::interfaces and include this, then select one with a single line, e.g.
 *        `using SelectedInterface = Spacetime::Interfaces::Standalone;`.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_INTERFACES_HPP
#define SPACETIME_INTERFACES_HPP

#include <spacetime/interface/InterfaceConcept.hpp>

#if defined(SPACETIME_INTERFACE_STANDALONE) && SPACETIME_INTERFACE_STANDALONE
#include <spacetime/interface/standalone/Interface.hpp>
#endif

#if defined(SPACETIME_INTERFACE_ROS2) && SPACETIME_INTERFACE_ROS2
#include <spacetime/interface/ros2/Interface.hpp>
#endif

#endif // SPACETIME_INTERFACES_HPP
