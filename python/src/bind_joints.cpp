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
                     py::arg("options") = JointOptions(),
                     "Joins prior A's robot at arclength `s_a` [m] to prior B's at `s_b`, each within its nodes' extent. "
                     "offset_covariance: the offset's uncertainty, 6x6 [rot, trans] for a rigid joint or 3x3 (translation) for a "
                     "spherical one, symmetric positive-definite. Raises ValueError for invalid inputs.")
                .def_readonly_static("joins_velocity", &JointT::joins_velocity, "Whether both robots' nodes have a velocity, so velocities are joined too.")
                .def("priorA", &JointT::priorA, "The prior of robot A.")
                .def("priorB", &JointT::priorB, "The prior of robot B.")
                .def("arclengthA", &JointT::arclengthA, "The joint's arclength on robot A [m].")
                .def("arclengthB", &JointT::arclengthB, "The joint's arclength on robot B [m].")
                .def("offset", &JointT::offset, "T_ab (JointOptions.offset).")
                .def("options", &JointT::options, "The joint's options.")
                .def("offsetCovariance", &JointT::offsetCovariance, "The offset's covariance, as given.")
                .def("hasOffset", &JointT::hasOffset, "Whether the joined points are apart (the offset has a translation).");
        }
    } // namespace

    void bindJoints(py::module_ &m)
    {
        py::enum_<JointType>(m, "JointType", "What a joint holds together.")
            .value("Rigid", JointType::Rigid, "The two points share their pose (and body twist), up to the offset.")
            .value("Spherical", JointType::Spherical, "The two points share their position (and linear velocity), up to the offset's translation; rotations are free.");

        py::class_<JointOptions>(m, "JointOptions", "Optional settings of a joint.")
            .def(py::init<>())
            .def_readwrite("type", &JointOptions::type, "Rigid (the default) or Spherical.")
            .def_readwrite("offset", &JointOptions::offset, "T_ab: prior B's point in prior A's point's frame (a spherical joint uses only its translation).")
            .def_readwrite("velocity_sigma_floor", &JointOptions::velocity_sigma_floor,
                           "Standard deviation (rad/s, m/s per component) of the velocity residual on top of the propagated offset uncertainty; > 0.");

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
            .def(py::init<const Pose &, const Vector6 &>(), py::arg("pose"), py::arg("varpi"), "Pose T_ib and velocity [rot, trans].")
            .def_readonly_static("dim", &EndEffectorValues::dim, "The node's tangent dimension (pose + velocity).")
            .def("pose", py::overload_cast<>(&EndEffectorValues::pose, py::const_), "The pose T_ib.")
            .def("pose", py::overload_cast<const Pose &>(&EndEffectorValues::pose), py::arg("value"), "Sets the pose T_ib.")
            .def("varpi", py::overload_cast<>(&EndEffectorValues::varpi, py::const_), "The generalized velocity, [rot, trans].")
            .def("varpi", py::overload_cast<const Vector6 &>(&EndEffectorValues::varpi), py::arg("value"), "Sets the generalized velocity, [rot, trans].");

        py::classh<EndEffectorNode, Spacetime::Types::NodeBase>(m, "EndEffectorNode", "An end effector node: pose ('p') and velocity ('v').")
            .def(py::init<KeyManager &, double, double>(), py::arg("km"), py::arg("arclength"), py::arg("time"),
                 "Registers the node's variables with `km`, at `time`.")
            .def("poseKey", &EndEffectorNode::poseKey, "The key of the pose variable.")
            .def("velocityKey", &EndEffectorNode::velocityKey, "The key of the velocity variable.");

        using Keys = Spacetime::Joints::EndEffectorKeys<Pose>;
        py::class_<Keys>(m, "EndEffectorKeys", "The keys a sensor's factor on an end effector is built on.")
            .def_static("Pose", &Keys::Pose, "The interpolated pose key.")
            .def_static("Velocity", &Keys::Velocity, "The interpolated velocity key.")
            .def_static("keys", &Keys::keys, "Every interpolated key of the node type.");

        bindContinuumRobotState<EndEffectorNode>(m, "EndEffectorState");

        py::classh<EndEffectorPrior, StatefulModel>(m, "EndEffectorPrior", "An end effector's WNOA (constant-velocity) prior in time, on a zero-length state.")
            .def(py::init(
                     [](std::shared_ptr<Spacetime::Types::State> state, const Pose &initial_pose, const Vector6 &q_psd_diag)
                     {
                         auto typed = std::dynamic_pointer_cast<EndEffectorState>(state);
                         if (!typed)
                             throw py::value_error("EndEffectorPrior: the state is not an end effector's (EndEffectorState).");
                         return std::make_shared<EndEffectorPrior>(typed, initial_pose, q_psd_diag);
                     }),
                 py::arg("state"), py::arg("initial_pose"), py::arg("q_psd_diag"),
                 "`state` is an EndEffectorState (else ValueError); initial_pose: the first row's initial guess (T_ib); "
                 "q_psd_diag: the diagonal of the WNOA power spectral density.");

        py::classh<EndEffector, StatefulModel>(
            m, "EndEffector", "A rigid body that robots attach to (attach, at any arclength), positioned only by them. Add it to the estimator once: "
            "it runs its prior and joints (do not add its prior separately). Sensors can measure it directly.")
            .def(py::init<const Pose &, const Vector6 &>(), py::arg("initial_pose"), py::arg("q_psd_diag") = Vector6::Ones().eval(),
                 "initial_pose: initial guess of its pose (T_ib); q_psd_diag: the diagonal of the WNOA power spectral density of its motion.")
            .def("attach", &EndEffector::attach<WnoaPrior>, py::arg("prior"), py::arg("s"), py::arg("offset_covariance"),
                 py::arg("options") = JointOptions(), "Joins the WNOA robot of `prior` at arclength `s` to this end effector (options.offset is T_point_ee); returns the "
                 "EndEffectorJoint. Raises ValueError for invalid inputs (see RobotJoint), keeping nothing.")
            .def("joints", &EndEffector::joints, "The joints attached so far, in attach() order.")
            .def("endEffectorPrior", &EndEffector::endEffectorPrior, "The prior extending the end effector's state.");
    }
} // namespace SpacetimePython
