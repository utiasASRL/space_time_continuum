/**
 * @file Types.hpp
 * @brief For downstream code (usually through the package-level <spacetime> header): the estimator's core types (models, states,
 *        measurements) and, when linking an interface (spacetime::interfaces, spacetime::standalone
 *        or spacetime::ros2), the estimator and Spacetime::Session / Spacetime::SessionOptions:
 *        @code
 *        Spacetime::SessionOptions options;
 *        Spacetime::Session session(options);                                    // the default (standalone) interface
 *        Spacetime::Session<Spacetime::INTERFACE_ROS2> ros(options, ros_options); // the ROS2 interface
 *        session.start(estimator);
 *        @endcode
 *        Library headers include the estimator/types/ headers they need instead of this one.
 */

#ifndef SPACETIME_TYPES_HPP
#define SPACETIME_TYPES_HPP

#include "estimator/types/InterpolatedKeys.hpp"
#include "estimator/types/Measurement.hpp"
#include "estimator/types/MeasurementQueue.hpp"
#include "estimator/types/Model.hpp"
#include "estimator/types/StatefulModel.hpp"
#include "estimator/types/ContinuumRobot.hpp"
#include "estimator/types/Robot.hpp"
#include "estimator/types/SensorBase.hpp"
#include "estimator/types/ActuatorState.hpp"
#include "estimator/types/ActuatorBase.hpp"
#include "estimator/types/States.hpp"

#if (defined(SPACETIME_INTERFACE_STANDALONE) && SPACETIME_INTERFACE_STANDALONE) || (defined(SPACETIME_INTERFACE_ROS2) && SPACETIME_INTERFACE_ROS2)
#include "estimator/Estimator.hpp"
#include "interface/Interfaces.hpp"

namespace Spacetime
{
    using Interface::Session;
    using Interface::SessionOptions;
    using Interface::INTERFACE_MUJOCO;
    using Interface::INTERFACE_ROS2;
    using Interface::INTERFACE_STANDALONE;
} // namespace Spacetime
#endif

#endif // SPACETIME_TYPES_HPP
