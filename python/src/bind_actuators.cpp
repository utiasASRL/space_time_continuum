/**
 * @file bind_actuators.cpp
 * @brief spacetime.actuators: the built-in actuators, for each prior they support, and their
 *        factors (for Python actuators to reuse).
 */

#include <spacetime/estimator/actuation_models/KinematicTendon.hpp>

#include "common.hpp"

namespace SpacetimePython
{
    namespace
    {
        using Spacetime::Types::ActuatorBase;
        template <typename PriorT>
        using KinematicTendon = Spacetime::Actuators::KinematicTendon<PriorT>;
        using Spacetime::Actuators::TendonLengthFactor;

        /// Binds KinematicTendon<PriorT> into `m` as `name`.
        template <typename PriorT>
        void bindKinematicTendon(py::module_ &m, const char *name, const char *robot)
        {
            using Tendon = KinematicTendon<PriorT>;
            py::classh<Tendon, ActuatorBase>(m, name, (std::string("A single tendon routed along a ") + robot + " robot (see actuators.KinematicTendon).").c_str())
                .def(py::init<std::shared_ptr<PriorT>, double, double, double, double>(), py::arg("prior"), py::arg("radius"), py::arg("angle"),
                     py::arg("termination"), py::arg("discretization"))
                .def(py::init<std::shared_ptr<PriorT>, double, double, std::vector<double>>(), py::arg("prior"), py::arg("radius"), py::arg("angle"),
                     py::arg("arclengths"))
                .def_property_readonly("prior", &Tendon::prior, "The robot's prior.")
                .def_property_readonly("radius", &Tendon::radius, "Distance of the routing points from the backbone [m].")
                .def_property_readonly("angle", &Tendon::angle, "Angle of the routing points about the backbone [rad], from body y towards z.")
                .def_property_readonly("termination", &Tendon::termination, "Arclength where the tendon ends [m].")
                .def_property_readonly("discretization", &Tendon::discretization, "Spacing of the routing disks [m]; None when they were given as arclengths.")
                .def_property_readonly("offset", &Tendon::offset, "The routing points' offset in each disk's body frame [m].")
                .def_property("lengthSigma", &Tendon::lengthSigma, &Tendon::setLengthSigma,
                              "Standard deviation of the tendon-length factor [m] (tight by default: the measurement's covariance sets the trust).")
                .def("predictedLength", &Tendon::predictedLength, py::arg("poses"), "The tendon length traced by `poses` (gtsam.Pose3), one per routing disk.")
                .def_static("diskPoseKey", &Tendon::diskPoseKey, py::arg("disk"), "The inner key of routing disk `disk`'s pose in getFactors' factor (the interpolated pose at point `disk`).");
        }

        /// make(tag) with a null pointer tag of the first of the priors PriorT, Rest... that `prior` is.
        template <typename PriorT, typename... Rest, typename Make>
        py::object dispatchOnPrior(const pStateful &prior, const char *actuator, Make &&make)
        {
            if (auto typed = std::dynamic_pointer_cast<PriorT>(prior))
                return make(typed);
            if constexpr (sizeof...(Rest) > 0)
                return dispatchOnPrior<Rest...>(prior, actuator, std::forward<Make>(make));
            else
                throw py::type_error(std::string(actuator) + ": no built-in actuator for this prior (write a Python ActuatorBase subclass instead).");
        }
    } // namespace

    void bindActuators(py::module_ &m)
    {
        bindKinematicTendon<WnoaPrior>(m, "KinematicTendonWNOA", "WNOA");
        bindKinematicTendon<WnovPrior>(m, "KinematicTendonWNOV", "WNOV");
        bindKinematicTendon<QuasiStaticPrior>(m, "KinematicTendonQuasiStatic", "quasi-static");

        py::class_<TendonLengthFactor, gtsam::NoiseModelFactor, std::shared_ptr<TendonLengthFactor>>(
            m, "TendonLengthFactor", "The error L - u: the straight-line path length through `offset` on every pose, minus the tendon length u (a 1-vector).")
            .def(py::init<const gtsam::KeyVector &, gtsam::Key, const gtsam::Point3 &, const gtsam::SharedNoiseModel &>(), py::arg("pose_keys"),
                 py::arg("length_key"), py::arg("offset"), py::arg("noise_model"))
            .def_static("pathLength", [](const std::vector<gtsam::Pose3> &poses, const gtsam::Point3 &offset)
                        { return TendonLengthFactor::pathLength(poses, offset); },
                        py::arg("poses"), py::arg("offset"), "The length of the straight-line path through `offset` carried by every pose.");

        m.def(
            "KinematicTendon",
            [](const pStateful &prior, double radius, double angle, double termination, double discretization)
            {
                return dispatchOnPrior<WnoaPrior, WnovPrior, QuasiStaticPrior>(prior, "KinematicTendon", [&](auto typed)
                                                                               {
                    using PriorT = typename decltype(typed)::element_type;
                    return py::cast(std::make_shared<KinematicTendon<PriorT>>(typed, radius, angle, termination, discretization)); });
            },
            py::arg("prior"), py::arg("radius"), py::arg("angle"), py::arg("termination"), py::arg("discretization"),
            "A single tendon routed along the robot (WNOA, WNOV or quasi-static) of `prior`, whose measurements are tendon lengths: "
            "each adds a tendon-length variable compared with the straight-line path through the routing disks traced by the "
            "state estimate. The disks sit at arclengths 0, discretization, 2 discretization, ... up to `termination` (always the "
            "last disk), each `radius` [m] from the backbone (body x) at `angle` [rad] from body y towards z. Raises TypeError for "
            "another prior, ValueError for an invalid radius, termination (past the robot) or discretization.");
        m.def(
            "KinematicTendon",
            [](const pStateful &prior, double radius, double angle, const std::vector<double> &arclengths)
            {
                return dispatchOnPrior<WnoaPrior, WnovPrior, QuasiStaticPrior>(prior, "KinematicTendon", [&](auto typed)
                                                                               {
                    using PriorT = typename decltype(typed)::element_type;
                    return py::cast(std::make_shared<KinematicTendon<PriorT>>(typed, radius, angle, arclengths)); });
            },
            py::arg("prior"), py::arg("radius"), py::arg("angle"), py::arg("arclengths"),
            "A single tendon routed through disks at the given `arclengths` [m] (at least two, strictly ascending, within the robot; "
            "the tendon ends at the last one), each `radius` [m] from the backbone at `angle` [rad]. Otherwise as the overload above.");
    }
} // namespace SpacetimePython
