/**
 * @file bind_joints.cpp
 * @brief spacetime.joints: joints between WNOA robots and end effectors.
 */

#include "bind_robot.hpp"
#include "common.hpp"

namespace SpacetimePython
{
    namespace
    {
        using Spacetime::Joints::JointOptions;
        using Spacetime::Joints::JointType;

        template <typename JointT, typename PriorA, typename PriorB>
        void bindJoint(py::module_ &m, const char *name, const char *doc)
        {
            py::classh<JointT, Spacetime::Types::Model>(m, name, doc)
                .def(py::init<std::shared_ptr<PriorA>, double, std::shared_ptr<PriorB>, double,
                              const Eigen::MatrixXd &, const JointOptions &>(),
                     py::arg("prior_a"), py::arg("s_a"), py::arg("prior_b"), py::arg("s_b"), py::arg("offset_covariance"),
                     py::arg("options") = JointOptions())
                .def_readonly_static("joins_velocity", &JointT::joins_velocity)
                .def("priorA", &JointT::priorA)
                .def("priorB", &JointT::priorB)
                .def("arclengthA", &JointT::arclengthA)
                .def("arclengthB", &JointT::arclengthB)
                .def("offset", &JointT::offset)
                .def("options", &JointT::options)
                .def("offsetCovariance", &JointT::offsetCovariance)
                .def("hasOffset", &JointT::hasOffset);
        }
    } // namespace

    void bindJoints(py::module_ &m)
    {
        py::enum_<JointType>(m, "JointType")
            .value("Rigid", JointType::Rigid, "The two points share their pose, up to the offset.")
            .value("Spherical", JointType::Spherical, "The two points share their position, up to the offset's translation.");

        py::class_<JointOptions>(m, "JointOptions")
            .def(py::init<>())
            .def_readwrite("type", &JointOptions::type)
            .def_readwrite("offset", &JointOptions::offset, "T_ab: prior B's point in prior A's point's frame.")
            .def_readwrite("velocity_sigma_floor", &JointOptions::velocity_sigma_floor);

        bindJoint<RobotJoint, WnoaPrior, WnoaPrior>(m, "RobotJoint", "A joint between the priors of two WNOA robots.");
        bindJoint<EndEffectorJoint, WnoaPrior, EndEffectorPrior>(m, "EndEffectorJoint", "A joint between a WNOA robot's prior (A) and an end effector's prior (B).");

        m.def(
            "Joint",
            [](std::shared_ptr<WnoaPrior> prior_a, double s_a, pStateful prior_b, double s_b, const Eigen::MatrixXd &offset_covariance,
               const JointOptions &options) -> py::object
            {
                if (auto end_effector = std::dynamic_pointer_cast<EndEffectorPrior>(prior_b))
                    return py::cast(std::make_shared<EndEffectorJoint>(prior_a, s_a, end_effector, s_b, offset_covariance, options));
                if (auto robot = std::dynamic_pointer_cast<WnoaPrior>(prior_b))
                    return py::cast(std::make_shared<RobotJoint>(prior_a, s_a, robot, s_b, offset_covariance, options));
                throw py::type_error("Joint: prior B must be a wnoa.Prior or an EndEffectorPrior.");
            },
            py::arg("prior_a"), py::arg("s_a"), py::arg("prior_b"), py::arg("s_b"), py::arg("offset_covariance"), py::arg("options") = JointOptions(),
            "Joins prior A (WNOA) at arclength s_a to prior B (WNOA, or an end effector's) at s_b: a RobotJoint or an EndEffectorJoint.");

        py::classh<EndEffectorValues, Spacetime::Types::NodeValues>(m, "EndEffectorValues", "An end effector node's pose and velocity (varpi).")
            .def(py::init<>())
            .def(py::init<const Pose &, const Vector6 &>(), py::arg("pose"), py::arg("varpi"))
            .def_readonly_static("dim", &EndEffectorValues::dim)
            .def("pose", py::overload_cast<>(&EndEffectorValues::pose, py::const_))
            .def("pose", py::overload_cast<const Pose &>(&EndEffectorValues::pose), py::arg("value"))
            .def("varpi", py::overload_cast<>(&EndEffectorValues::varpi, py::const_))
            .def("varpi", py::overload_cast<const Vector6 &>(&EndEffectorValues::varpi), py::arg("value"));

        py::classh<EndEffectorNode, Spacetime::Types::NodeBase>(m, "EndEffectorNode", "An end effector node: pose ('p') and velocity ('v').")
            .def(py::init<KeyManager &, double, double>(), py::arg("km"), py::arg("arclength"), py::arg("time"))
            .def("poseKey", &EndEffectorNode::poseKey)
            .def("velocityKey", &EndEffectorNode::velocityKey);

        using Keys = Spacetime::Joints::EndEffectorKeys<Pose>;
        py::class_<Keys>(m, "EndEffectorKeys", "The keys a sensor's factor on an end effector is built on.")
            .def_static("Pose", &Keys::Pose)
            .def_static("Velocity", &Keys::Velocity)
            .def_static("keys", &Keys::keys);

        bindContinuumRobotState<EndEffectorNode>(m, "EndEffectorState");

        py::classh<EndEffectorPrior, StatefulModel>(m, "EndEffectorPrior", "An end effector's constant-velocity prior in time.")
            .def(py::init(
                     [](std::shared_ptr<Spacetime::Types::State> state, const Pose &initial_pose, const Vector6 &q_psd_diag)
                     {
                         auto typed = std::dynamic_pointer_cast<EndEffectorState>(state);
                         if (!typed)
                             throw py::value_error("EndEffectorPrior: the state is not an end effector's (EndEffectorState).");
                         return std::make_shared<EndEffectorPrior>(typed, initial_pose, q_psd_diag);
                     }),
                 py::arg("state"), py::arg("initial_pose"), py::arg("q_psd_diag"));

        py::classh<EndEffector, StatefulModel>(
            m, "EndEffector", "A rigid body joined to the tips of robots (attach); add it to the estimator once: it runs its prior and joints.")
            .def(py::init<const Pose &, const Vector6 &>(), py::arg("initial_pose"), py::arg("q_psd_diag") = Vector6::Ones().eval())
            .def("attach", &EndEffector::attach<WnoaPrior>, py::arg("prior"), py::arg("s"), py::arg("offset_covariance"),
                 py::arg("options") = JointOptions(), "Joins the WNOA robot of `prior` at arclength `s` to this end effector; returns the EndEffectorJoint.")
            .def("joints", &EndEffector::joints)
            .def("endEffectorPrior", &EndEffector::endEffectorPrior);
    }
} // namespace SpacetimePython
