/**
 * @file bind_joints.cpp
 * @brief spacetime.joints: joints between robots (of any built-in prior) and end effectors.
 */

#include "bind_robot.hpp"
#include "common.hpp"

namespace SpacetimePython
{
    namespace
    {
        using Spacetime::Joints::JointOptions;
        using Spacetime::Joints::JointType;

        /// The robot priors' Python names, for the TypeErrors.
        constexpr const char *kRobotPriorNames = "a wnoa, wnov or wnoa_quasi_static Prior";

        /// Joint<A, B> between `prior_a` (a robot) and `prior_b` (a robot or an end effector), whatever their types.
        std::shared_ptr<JointBase> makeJoint(const pStateful &prior_a, double s_a, const pStateful &prior_b, double s_b,
                                             const Eigen::MatrixXd &offset_covariance, const JointOptions &options)
        {
            std::shared_ptr<JointBase> joint;
            bool b_known = false;
            const bool a_known = visitPrior(RobotPriors{}, prior_a, [&](auto typed_a)
                                            { b_known = visitPrior(JointSideBPriors{}, prior_b, [&](auto typed_b)
                                                                   {
                                                                       using A = typename decltype(typed_a)::element_type;
                                                                       using B = typename decltype(typed_b)::element_type;
                                                                       joint = std::make_shared<Spacetime::Joints::Joint<A, B>>(typed_a, s_a, typed_b, s_b, offset_covariance, options);
                                                                   }); });
            if (!a_known)
                throw py::type_error(std::string("Joint: prior A must be ") + kRobotPriorNames + ".");
            if (!b_known)
                throw py::type_error(std::string("Joint: prior B must be ") + kRobotPriorNames + ", or an EndEffectorPrior.");
            return joint;
        }

        /**
         * Python overrides of a JointBase. A subclass constructed with the joint's arguments wraps the
         * built-in Joint<A, B> they select, and every method it does not override (including the Model
         * hooks, e.g. getNewFactors) forwards to it. Constructed without arguments, it wraps nothing: it
         * must override the accessors (stateA/stateB default to its priors' states, hasOffset to the
         * offset's translation, joinsVelocity to False) and produce its own factors.
         */
        struct PyJoint : JointBase, py::trampoline_self_life_support
        {
            PyJoint() = default;
            explicit PyJoint(std::shared_ptr<JointBase> inner) : m_inner(std::move(inner)) {}

            Factors extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override
            {
                SPACETIME_OVERRIDE(Factors, JointBase, "extendState", ref(km), time, ref(current_values), ref(new_values));
                return m_inner ? m_inner->extendState(km, time, current_values, new_values) : JointBase::extendState(km, time, current_values, new_values);
            }

            Factors getNewFactors(double time, gtsam::Values &new_values) override
            {
                SPACETIME_OVERRIDE(Factors, JointBase, "getNewFactors", time, ref(new_values));
                return m_inner ? m_inner->getNewFactors(time, new_values) : JointBase::getNewFactors(time, new_values);
            }

            void onMarginalization(double cutoff, const gtsam::Values &values) override
            {
                SPACETIME_OVERRIDE(void, JointBase, "onMarginalization", cutoff, ref(values));
                m_inner ? m_inner->onMarginalization(cutoff, values) : JointBase::onMarginalization(cutoff, values);
            }

            const gtsam::Values &lockedValues() const override
            {
                if (cachedOverride("lockedValues", m_locked_values))
                    return m_locked_values;
                return m_inner ? m_inner->lockedValues() : JointBase::lockedValues();
            }

            void resolveLocks(const gtsam::Values &new_values) override
            {
                m_inner ? m_inner->resolveLocks(new_values) : JointBase::resolveLocks(new_values);
            }

            std::vector<gtsam::KeyVector> covarianceKeyGroups(const gtsam::Values &values) const override
            {
                SPACETIME_OVERRIDE(std::vector<gtsam::KeyVector>, JointBase, "covarianceKeyGroups", ref(values));
                return m_inner ? m_inner->covarianceKeyGroups(values) : JointBase::covarianceKeyGroups(values);
            }

            std::shared_ptr<StatefulModel> priorModelA() const override
            {
                SPACETIME_OVERRIDE(std::shared_ptr<StatefulModel>, JointBase, "priorA");
                return inner("priorA")->priorModelA();
            }

            std::shared_ptr<StatefulModel> priorModelB() const override
            {
                SPACETIME_OVERRIDE(std::shared_ptr<StatefulModel>, JointBase, "priorB");
                return inner("priorB")->priorModelB();
            }

            std::shared_ptr<const Spacetime::Types::State> stateA() const override
            {
                SPACETIME_OVERRIDE(std::shared_ptr<const Spacetime::Types::State>, JointBase, "stateA");
                return m_inner ? m_inner->stateA() : stateOf(priorModelA());
            }

            std::shared_ptr<const Spacetime::Types::State> stateB() const override
            {
                SPACETIME_OVERRIDE(std::shared_ptr<const Spacetime::Types::State>, JointBase, "stateB");
                return m_inner ? m_inner->stateB() : stateOf(priorModelB());
            }

            double arclengthA() const override
            {
                SPACETIME_OVERRIDE(double, JointBase, "arclengthA");
                return inner("arclengthA")->arclengthA();
            }

            double arclengthB() const override
            {
                SPACETIME_OVERRIDE(double, JointBase, "arclengthB");
                return inner("arclengthB")->arclengthB();
            }

            const gtsam::Pose3 &offset() const override
            {
                if (cachedOverride("offset", m_offset))
                    return m_offset;
                return inner("offset")->offset();
            }

            const JointOptions &options() const override
            {
                if (cachedOverride("options", m_options))
                    return m_options;
                return inner("options")->options();
            }

            const Eigen::MatrixXd &offsetCovariance() const override
            {
                if (cachedOverride("offsetCovariance", m_offset_covariance))
                    return m_offset_covariance;
                return inner("offsetCovariance")->offsetCovariance();
            }

            bool hasOffset() const override
            {
                SPACETIME_OVERRIDE(bool, JointBase, "hasOffset");
                return m_inner ? m_inner->hasOffset() : offset().translation().norm() > 1e-9;
            }

            bool joinsVelocity() const override
            {
                SPACETIME_OVERRIDE(bool, JointBase, "joinsVelocity");
                return m_inner ? m_inner->joinsVelocity() : false;
            }

        private:
            /// The built-in joint this one extends, or null for a joint defined wholly in Python.
            std::shared_ptr<JointBase> m_inner;
            // A Python override's results are kept here, since the callers receive references.
            mutable gtsam::Values m_locked_values;
            mutable gtsam::Pose3 m_offset;
            mutable JointOptions m_options;
            mutable Eigen::MatrixXd m_offset_covariance;

            /// Stores the Python override `name`'s result in `cache`; false if it is not overridden.
            template <typename T>
            bool cachedOverride(const char *name, T &cache) const
            {
                py::gil_scoped_acquire gil;
                if (py::function py_override = py::get_override(static_cast<const JointBase *>(this), name))
                {
                    cache = py_override().template cast<T>();
                    return true;
                }
                return false;
            }

            /// The wrapped joint; throws for a joint defined wholly in Python that did not override `name`.
            const JointBase *inner(const char *name) const
            {
                if (!m_inner)
                    throw pureVirtual(std::string("Joint.") + name);
                return m_inner.get();
            }

            static std::shared_ptr<const Spacetime::Types::State> stateOf(const std::shared_ptr<StatefulModel> &prior)
            {
                if (!prior)
                    throw std::runtime_error("Joint: the prior is None.");
                return prior->state();
            }
        };
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

        py::classh<JointBase, PyJoint, Spacetime::Types::Model>(
            m, "Joint",
            "A joint holding the point at arclength s_a of robot A and the one at s_b of robot B together, by soft factors on their "
            "poses (or positions) and, when both robots' nodes have velocities, their velocities. Robot A is given by its prior "
            "(wnoa, wnov or wnoa_quasi_static), robot B by its prior or an end effector's. Add it to the estimator after both "
            "robots (an end effector adds its own joints).\n\n"
            "Subclassing: a subclass that calls super().__init__ with the joint's arguments extends the built-in joint they select: "
            "every method it does not override forwards to it, e.g. getNewFactors can add factors to super().getNewFactors(...). "
            "One that calls super().__init__() defines a joint wholly in Python: it overrides priorA, priorB, arclengthA, arclengthB, "
            "offset, options, offsetCovariance and its Model hooks (getNewFactors, ...); stateA/stateB default to its priors' states, "
            "hasOffset to whether offset() has a translation, joinsVelocity to False. The visualizer and rviz draw either kind.")
            .def(py::init(&makeJoint, [](const pStateful &prior_a, double s_a, const pStateful &prior_b, double s_b,
                                         const Eigen::MatrixXd &offset_covariance, const JointOptions &options)
                          { return std::make_shared<PyJoint>(makeJoint(prior_a, s_a, prior_b, s_b, offset_covariance, options)); }),
                 py::arg("prior_a"), py::arg("s_a"), py::arg("prior_b"), py::arg("s_b"), py::arg("offset_covariance"),
                 py::arg("options") = JointOptions(),
                 "Joins prior A's robot at arclength `s_a` [m] to prior B's at `s_b`, each within its nodes' extent. "
                 "offset_covariance: the offset's uncertainty, 6x6 [rot, trans] for a rigid joint or 3x3 (translation) for a "
                 "spherical one, symmetric positive-definite. Raises TypeError for an unsupported prior, ValueError for invalid inputs.")
            .def(py::init([]() -> std::shared_ptr<JointBase>
                          { throw py::type_error("Joint(): only a subclass defining its joint in Python may be constructed without arguments."); },
                          []() { return std::make_shared<PyJoint>(); }),
                 "For a subclass that defines its joint wholly in Python (see the class documentation).")
            .def_property_readonly("joins_velocity", &JointBase::joinsVelocity,
                                   "Whether both robots' nodes have a velocity (not quasi-static), so velocities are joined too (see joinsVelocity).")
            .def("joinsVelocity", &JointBase::joinsVelocity, "As joins_velocity; the method a subclass overrides.")
            .def("stateA", &JointBase::stateA, "Robot A's state (by default its prior's).")
            .def("stateB", &JointBase::stateB, "Robot B's state (by default its prior's).")
            .def("priorA", &JointBase::priorModelA, "The prior of robot A.")
            .def("priorB", &JointBase::priorModelB, "The prior of robot B (or of the end effector).")
            .def("arclengthA", &JointBase::arclengthA, "The joint's arclength on robot A [m].")
            .def("arclengthB", &JointBase::arclengthB, "The joint's arclength on robot B [m].")
            .def("offset", &JointBase::offset, "T_ab (JointOptions.offset).")
            .def("options", &JointBase::options, "The joint's options.")
            .def("offsetCovariance", &JointBase::offsetCovariance, "The offset's covariance, as given.")
            .def("hasOffset", &JointBase::hasOffset, "Whether the joined points are apart (the offset has a translation).");

        py::classh<EndEffectorNodeValues, Spacetime::Types::NodeValuesBase>(m, "EndEffectorNodeValues", "An end effector node's pose and velocity (varpi).")
            .def(py::init<>())
            .def(py::init<const Pose &, const Vector6 &>(), py::arg("pose"), py::arg("varpi"), "Pose T_ib and velocity [rot, trans].")
            .def_readonly_static("dim", &EndEffectorNodeValues::dim, "The node's tangent dimension (pose + velocity).")
            .def("pose", py::overload_cast<>(&EndEffectorNodeValues::pose, py::const_), "The pose T_ib.")
            .def("pose", py::overload_cast<const Pose &>(&EndEffectorNodeValues::pose), py::arg("value"), "Sets the pose T_ib.")
            .def("varpi", py::overload_cast<>(&EndEffectorNodeValues::varpi, py::const_), "The generalized velocity, [rot, trans].")
            .def("varpi", py::overload_cast<const Vector6 &>(&EndEffectorNodeValues::varpi), py::arg("value"), "Sets the generalized velocity, [rot, trans].");

        py::classh<EndEffectorNode, Spacetime::Types::NodeBase>(m, "EndEffectorNode", "An end effector node: pose ('p') and velocity ('v').")
            .def(py::init<KeyManager &, double, double>(), py::arg("km"), py::arg("arclength"), py::arg("time"),
                 "Registers the node's variables with `km`, at `time`.")
            .def("poseKey", &EndEffectorNode::poseKey, "The key of the pose variable.")
            .def("velocityKey", &EndEffectorNode::velocityKey, "The key of the velocity variable.");

        using Keys = Spacetime::Joints::EndEffectorKeys<Pose>;
        py::class_<Keys>(m, "EndEffectorKeys", "The keys a sensor's factor on an end effector is built on.")
            .def_static("Pose", &pointOptional<gtsam::Key, &Keys::Pose, &Keys::Pose>, py::arg("point") = py::none(), "The interpolated pose key. `point` selects the key at the point-th arclength of a multi-point factor (see types.InterpolatedKeyAt).")
            .def_static("Velocity", &pointOptional<gtsam::Key, &Keys::Velocity, &Keys::Velocity>, py::arg("point") = py::none(), "The interpolated velocity key. `point` selects the key at the point-th arclength of a multi-point factor (see types.InterpolatedKeyAt).")
            .def_static("keys", &pointOptional<gtsam::KeyVector, &Keys::keys, &Keys::keys>, py::arg("point") = py::none(), "Every interpolated key of the node type. `point` selects the key at the point-th arclength of a multi-point factor (see types.InterpolatedKeyAt).");

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
                 "q_psd_diag: the diagonal of the WNOA power spectral density.")
            .def(py::init(
                     [](std::shared_ptr<Spacetime::Types::State> state, const Pose &initial_pose)
                     {
                         auto typed = std::dynamic_pointer_cast<EndEffectorState>(state);
                         if (!typed)
                             throw py::value_error("EndEffectorPrior: the state is not an end effector's (EndEffectorState).");
                         return std::make_shared<EndEffectorPrior>(typed, initial_pose);
                     }),
                 py::arg("state"), py::arg("initial_pose"), "An end-effector prior with the default q_psd_diag (EndEffectorPrior.DefaultCovariances()).")
            .def_static("DefaultCovariances", &EndEffectorPrior::DefaultCovariances, "The default q_psd_diag: ones.");

        py::classh<EndEffector, StatefulModel>(
            m, "EndEffector", "A rigid body that robots attach to (attach, at any arclength), positioned only by them. Add it to the estimator once: "
            "it runs its prior and joints (do not add its prior separately). Sensors can measure it directly.")
            .def(py::init<const Pose &, const Vector6 &>(), py::arg("initial_pose"), py::arg("q_psd_diag") = Vector6::Ones().eval(),
                 "initial_pose: initial guess of its pose (T_ib); q_psd_diag: the diagonal of the WNOA power spectral density of its motion.")
            .def(
                "attach",
                [](EndEffector &self, const pStateful &prior, double s, const Eigen::MatrixXd &offset_covariance, const JointOptions &options)
                {
                    std::shared_ptr<JointBase> joint;
                    if (!visitPrior(RobotPriors{}, prior, [&](auto typed)
                                    { joint = self.attach(typed, s, offset_covariance, options); }))
                        throw py::type_error(std::string("EndEffector.attach: the prior must be ") + kRobotPriorNames + ".");
                    return joint;
                },
                py::arg("prior"), py::arg("s"), py::arg("offset_covariance"), py::arg("options") = JointOptions(),
                "Joins the robot of `prior` (wnoa, wnov or wnoa_quasi_static) at arclength `s` to this end effector (options.offset is "
                "T_point_ee); returns the Joint. Raises TypeError for an unsupported prior, ValueError for invalid inputs (see Joint), "
                "keeping nothing.")
            .def("joints", &EndEffector::joints, "The joints attached so far, in attach() order.")
            .def("endEffectorPrior", &EndEffector::endEffectorPrior, "The prior extending the end effector's state.");
    }
} // namespace SpacetimePython
