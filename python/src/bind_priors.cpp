/**
 * @file bind_priors.cpp
 * @brief spacetime.priors.wnoa, .wnov and .wnoa_quasi_static: the GP priors on gtsam.Pose3, their
 *        node values, nodes, interpolated keys and grid states.
 */

#include "bind_robot.hpp"
#include "common.hpp"

namespace SpacetimePython
{
    namespace
    {
        // The factories return raw pointers: pybind11's smart_holder then owns the prior, which it
        // needs to hand a Python subclass of one (enable_shared_from_this) to C++ as a shared_ptr.

        /// The robot state `state` as the grid state of prior PriorT. @throws py::value_error if it is not one.
        template <typename PriorT>
        std::shared_ptr<typename PriorT::StateType> gridState(const std::shared_ptr<Spacetime::Types::State> &state, const char *prior_name)
        {
            auto typed = std::dynamic_pointer_cast<typename PriorT::StateType>(state);
            if (!typed)
                throw py::value_error(std::string(prior_name) + ": the state is not one of its grid states (use its module's ContinuumRobotState).");
            return typed;
        }

        /// A grid prior's InterpolationFactor (the graph's measurement factors) and its Prior.queryJacobians.
        template <typename PriorT, typename PriorClass>
        void bindGridPrior(py::module_ &m, PriorClass &prior)
        {
            using Factor = Spacetime::Priors::InterpolationFactor<typename PriorT::InterpolatorType>;
            py::class_<Factor, gtsam::NoiseModelFactor, std::shared_ptr<Factor>>(
                m, "InterpolationFactor",
                "A measurement's inner factor on the node interpolated at (time, arclength) from its bordering nodes. "
                "noiseModel() is the inner one; unless fixedNoiseModel(), linearize() whitens with it plus G Pcheck G^T "
                "(G = innerJacobian, Pcheck = the conditional covariance of Prior.queryCovariance).")
                .def("innerFactor", &Factor::innerFactor, "The measurement's factor on the interpolated keys, with the base noise model.")
                .def("time", &Factor::time, "The query time [s].")
                .def("arclength", &Factor::arclength, "The query arclength [m].")
                .def("fixedNoiseModel", &Factor::fixedNoiseModel, "Whether the noise model is the inner one, without the interpolation's conditional covariance.")
                .def("innerJacobian", &Factor::innerJacobian, py::arg("values"),
                     "The inner factor's error Jacobian w.r.t. the interpolated node (error dim x node dim).");

            prior.def(
                "queryJacobians",
                [](const PriorT &self, double time, double arclength, const gtsam::Values &values)
                {
                    std::unordered_map<gtsam::Key, Eigen::MatrixXd> jacobians;
                    auto node = self.queryJacobians(time, arclength, values, jacobians);
                    return py::make_tuple(node, jacobians);
                },
                py::arg("time"), py::arg("arclength"), py::arg("values"),
                "(node values, {corner key: node dim x key dim Jacobian}) at (time, arclength); (None, {}) where query is None.");
        }

        void bindWnoa(py::module_ &m)
        {
            namespace WNOA = Spacetime::Priors::WNOA;
            using MatrixN = WnoaPrior::MatrixN;
            using Matrix3N = WnoaPrior::Matrix3N;

            py::enum_<WNOA::FactorType>(m, "FactorType", "Which GP prior factors connect neighbouring nodes.")
                .value("WNOA", WNOA::FactorType::WNOA, "The binary time and space factors on the group, as in the TRO paper (Eq. 20); the default.")
                .value("Legacy", WNOA::FactorType::Legacy, "The same group factors, with the legacy code's strain-velocity coupling term.")
                .value("Experimental", WNOA::FactorType::Experimental, "gtsam.WnoaMotionFactor + BetweenFactor + PriorFactor: the GP on the Lie algebra.");

            py::class_<WNOA::PriorOptions>(m, "PriorOptions", "Options of the WNOA prior (see utilities.ConfigLoader.loadWNOAPriorOptions).")
                .def(py::init<>())
                .def_readwrite("factor_type", &WNOA::PriorOptions::factor_type, "Which GP prior factors connect neighbouring nodes.")
                .def_readwrite("fixed_noise_model", &WNOA::PriorOptions::fixed_noise_model,
                               "Use each measurement's own noise model for its InterpolationFactor, without the interpolation's conditional covariance.")
                .def_readwrite("cache_noise_model", &WNOA::PriorOptions::cache_noise_model,
                               "Let each InterpolationFactor's error() reuse the augmented noise model of its latest linearization.")
                .def_readwrite("remove_unary", &WNOA::PriorOptions::remove_unary, "Add the unary factor to the first row's base node only, not to every row's.");

            py::classh<WnoaValues, Spacetime::Types::NodeValues>(m, "Values", "A WNOA node's pose, strain (epsilon) and velocity (varpi), [rot, trans].")
                .def(py::init<>(), "Identity pose, zero strain and velocity.")
                .def_readonly_static("dim", &WnoaValues::dim, "The node's tangent dimension (pose + strain + velocity).")
                .def("pose", py::overload_cast<>(&WnoaValues::pose, py::const_), "The pose T_ib.")
                .def("pose", py::overload_cast<const Pose &>(&WnoaValues::pose), py::arg("value"), "Sets the pose T_ib.")
                .def("epsilon", py::overload_cast<>(&WnoaValues::epsilon, py::const_), "The generalized strain, [rot, trans].")
                .def("epsilon", py::overload_cast<const Vector6 &>(&WnoaValues::epsilon), py::arg("value"), "Sets the generalized strain, [rot, trans].")
                .def("varpi", py::overload_cast<>(&WnoaValues::varpi, py::const_), "The generalized velocity, [rot, trans].")
                .def("varpi", py::overload_cast<const Vector6 &>(&WnoaValues::varpi), py::arg("value"), "Sets the generalized velocity, [rot, trans].");

            py::classh<WnoaNode, Spacetime::Types::NodeBase>(m, "Node", "A WNOA node: pose ('p'), strain ('s') and velocity ('v') variables.")
                .def(py::init<KeyManager &, double, double>(), py::arg("km"), py::arg("arclength"), py::arg("time"),
                     "Registers the node's variables with `km`, at `time`.")
                .def("poseKey", &WnoaNode::poseKey, "The key of the pose variable.")
                .def("strainKey", &WnoaNode::strainKey, "The key of the strain variable.")
                .def("velocityKey", &WnoaNode::velocityKey, "The key of the velocity variable.")
                .def("getValues", &WnoaNode::getValues, py::arg("values"), "The node's Values read from `values`; raises if a variable is absent.")
                .def("setValues", &WnoaNode::setValues, py::arg("values"), py::arg("node_values"), "Inserts `node_values` into `values` at the node's keys.");

            using Keys = WNOA::InterpolatedKeys<Pose>;
            py::class_<Keys>(m, "InterpolatedKeys", "The keys a sensor's factor is built on, resolved onto the prior's nodes.")
                .def_static("Pose", &Keys::Pose, "The interpolated pose key.")
                .def_static("Strain", &Keys::Strain, "The interpolated strain key.")
                .def_static("Velocity", &Keys::Velocity, "The interpolated velocity key.")
                .def_static("keys", &Keys::keys, "Every interpolated key of the node type.");

            bindContinuumRobotState<WnoaNode>(m, "ContinuumRobotState");

            const auto make = [](auto tag)
            {
                using T = typename decltype(tag)::type;
                return [](std::shared_ptr<Spacetime::Types::State> state, const WnoaValues &base_start_point, const Matrix3N &Q0,
                          const MatrixN &Q1, const MatrixN &Q2, const MatrixN &Q3, const MatrixN &Q4, const WNOA::PriorOptions &options)
                { return new T(gridState<WnoaPrior>(state, "wnoa.Prior"), base_start_point, Q0, Q1, Q2, Q3, Q4, options); };
            };
            auto prior = py::classh<WnoaPrior, PyStatefulModel<WnoaPrior>, StatefulModel>(m, "Prior",
                                                                            "The white-noise-on-acceleration GP prior over a robot's grid state.")
                .def(py::init(make(std::type_identity<WnoaPrior>()), make(std::type_identity<PyStatefulModel<WnoaPrior>>())), py::arg("state"),
                     py::arg("base_start_point"), py::arg("Q0") = Matrix3N::Identity().eval(),
                     py::arg("Q1") = MatrixN::Identity().eval(), py::arg("Q2") = MatrixN::Identity().eval(), py::arg("Q3") = MatrixN::Identity().eval(),
                     py::arg("Q4") = MatrixN::Identity().eval(), py::arg("options") = WNOA::PriorOptions(),
                     "`state` is a wnoa.ContinuumRobotState (else ValueError). base_start_point: the unary factor's base-node values "
                     "(pose T_ib, strain and velocity [rot, trans]). Q0: unary covariance over [pose, strain, velocity]. "
                     "Q1: time-axis (pose, velocity) PSD. Q2: arclength-axis (pose, strain) PSD. Q3: strain random walk in time PSD. "
                     "Q4: velocity random walk in arclength PSD. Q1..Q4 are taken as diagonal.")
                .def("options", &WnoaPrior::options, py::return_value_policy::reference_internal, "The prior's options.");
            bindGridPrior<WnoaPrior>(m, prior);
        }

        void bindWnov(py::module_ &m)
        {
            namespace WNOV = Spacetime::Priors::WNOV;
            using MatrixN = WnovPrior::MatrixN;

            py::class_<WNOV::PriorOptions>(m, "PriorOptions", "Options of the WNOV prior (see utilities.ConfigLoader.loadWNOVPriorOptions).")
                .def(py::init<>())
                .def_readwrite("fixed_noise_model", &WNOV::PriorOptions::fixed_noise_model,
                               "Use each measurement's own noise model for its InterpolationFactor, without the interpolation's conditional covariance.")
                .def_readwrite("cache_noise_model", &WNOV::PriorOptions::cache_noise_model,
                               "Let each InterpolationFactor's error() reuse the augmented noise model of its latest linearization.")
                .def_readwrite("remove_unary", &WNOV::PriorOptions::remove_unary, "Add the unary factor to the first row's base node only, not to every row's.");

            py::classh<WnovValues, Spacetime::Types::NodeValues>(m, "Values", "A WNOV node's pose.")
                .def(py::init<>(), "Identity pose.")
                .def(py::init<const Pose &>(), py::arg("value"), "Pose `value` (T_ib).")
                .def_readonly_static("dim", &WnovValues::dim, "The node's tangent dimension (the pose's).")
                .def("pose", py::overload_cast<>(&WnovValues::pose, py::const_), "The pose T_ib.")
                .def("pose", py::overload_cast<const Pose &>(&WnovValues::pose), py::arg("value"), "Sets the pose T_ib.");

            py::classh<WnovNode, Spacetime::Types::NodeBase>(m, "Node", "A WNOV node: a pose ('p') variable.")
                .def(py::init<KeyManager &, double, double>(), py::arg("km"), py::arg("arclength"), py::arg("time"),
                     "Registers the node's variables with `km`, at `time`.")
                .def("poseKey", &WnovNode::poseKey, "The key of the pose variable.")
                .def("getValues", &WnovNode::getValues, py::arg("values"), "The node's Values read from `values`; raises if a variable is absent.")
                .def("setValues", &WnovNode::setValues, py::arg("values"), py::arg("node_values"), "Inserts `node_values` into `values` at the node's keys.");

            using Keys = WNOV::InterpolatedKeys<Pose>;
            py::class_<Keys>(m, "InterpolatedKeys", "The keys a sensor's factor is built on, resolved onto the prior's nodes.")
                .def_static("Pose", &Keys::Pose, "The interpolated pose key.")
                .def_static("keys", &Keys::keys, "Every interpolated key of the node type.");

            bindContinuumRobotState<WnovNode>(m, "ContinuumRobotState");

            const auto make = [](auto tag)
            {
                using T = typename decltype(tag)::type;
                return [](std::shared_ptr<Spacetime::Types::State> state, const Pose &T0, const Vector6 &epsilon0, const MatrixN &Q0,
                          const MatrixN &Q1, const MatrixN &Q2, const WNOV::PriorOptions &options)
                { return new T(gridState<WnovPrior>(state, "wnov.Prior"), T0, epsilon0, Q0, Q1, Q2, options); };
            };
            auto prior = py::classh<WnovPrior, PyStatefulModel<WnovPrior>, StatefulModel>(m, "Prior",
                                                                            "The white-noise-on-velocity GP prior over a robot's grid state.")
                .def(py::init(make(std::type_identity<WnovPrior>()), make(std::type_identity<PyStatefulModel<WnovPrior>>())), py::arg("state"), py::arg("T0"),
                     py::arg("epsilon0"), py::arg("Q0") = MatrixN::Identity().eval(),
                     py::arg("Q1") = MatrixN::Identity().eval(), py::arg("Q2") = MatrixN::Identity().eval(), py::arg("options") = WNOV::PriorOptions(),
                     "`state` is a wnov.ContinuumRobotState (else ValueError). T0: the base pose T_ib the unary factor holds each row's base node to. "
                     "epsilon0: nominal strain [rot, trans]. Q0: unary covariance. Q1: arclength random walk PSD (per metre). "
                     "Q2: time random walk PSD (per second).")
                .def("options", &WnovPrior::options, py::return_value_policy::reference_internal, "The prior's options.");
            bindGridPrior<WnovPrior>(m, prior);
        }

        void bindQuasiStatic(py::module_ &m)
        {
            namespace QS = Spacetime::Priors::WNOAQuasiStatic;
            using MatrixN = QuasiStaticPrior::MatrixN;
            using Matrix2N = QuasiStaticPrior::Matrix2N;

            py::class_<QS::PriorOptions>(m, "PriorOptions", "Options of the quasi-static prior (see utilities.ConfigLoader.loadWNOAQuasiStaticPriorOptions).")
                .def(py::init<>())
                .def_readwrite("fixed_noise_model", &QS::PriorOptions::fixed_noise_model,
                               "Use each measurement's own noise model for its InterpolationFactor, without the interpolation's conditional covariance.")
                .def_readwrite("cache_noise_model", &QS::PriorOptions::cache_noise_model,
                               "Let each InterpolationFactor's error() reuse the augmented noise model of its latest linearization.");

            py::classh<QuasiStaticValues, Spacetime::Types::NodeValues>(m, "Values", "A quasi-static node's pose and strain (epsilon), [rot, trans].")
                .def(py::init<>(), "Identity pose, zero strain.")
                .def(py::init<const Pose &, const Vector6 &>(), py::arg("pose"), py::arg("epsilon"), "Pose T_ib and strain [rot, trans].")
                .def_readonly_static("dim", &QuasiStaticValues::dim, "The node's tangent dimension (pose + strain).")
                .def("pose", py::overload_cast<>(&QuasiStaticValues::pose, py::const_), "The pose T_ib.")
                .def("pose", py::overload_cast<const Pose &>(&QuasiStaticValues::pose), py::arg("value"), "Sets the pose T_ib.")
                .def("epsilon", py::overload_cast<>(&QuasiStaticValues::epsilon, py::const_), "The generalized strain, [rot, trans].")
                .def("epsilon", py::overload_cast<const Vector6 &>(&QuasiStaticValues::epsilon), py::arg("value"), "Sets the generalized strain, [rot, trans].");

            py::classh<QuasiStaticNode, Spacetime::Types::NodeBase>(m, "Node", "A quasi-static node: pose ('p') and strain ('s') variables.")
                .def(py::init<KeyManager &, double, double>(), py::arg("km"), py::arg("arclength"), py::arg("time"),
                     "Registers the node's variables with `km`, at `time`.")
                .def("poseKey", &QuasiStaticNode::poseKey, "The key of the pose variable.")
                .def("strainKey", &QuasiStaticNode::strainKey, "The key of the strain variable.")
                .def("getValues", &QuasiStaticNode::getValues, py::arg("values"), "The node's Values read from `values`; raises if a variable is absent.")
                .def("setValues", &QuasiStaticNode::setValues, py::arg("values"), py::arg("node_values"), "Inserts `node_values` into `values` at the node's keys.");

            using Keys = QS::InterpolatedKeys<Pose>;
            py::class_<Keys>(m, "InterpolatedKeys", "The keys a sensor's factor is built on, resolved onto the prior's nodes.")
                .def_static("Pose", &Keys::Pose, "The interpolated pose key.")
                .def_static("Strain", &Keys::Strain, "The interpolated strain key.")
                .def_static("keys", &Keys::keys, "Every interpolated key of the node type.");

            bindContinuumRobotState<QuasiStaticNode>(m, "ContinuumRobotState");

            const auto make = [](auto tag)
            {
                using T = typename decltype(tag)::type;
                return [](std::shared_ptr<Spacetime::Types::State> state, const QuasiStaticValues &base_start_point, const Matrix2N &Q0,
                          const MatrixN &Q2, const QS::PriorOptions &options)
                { return new T(gridState<QuasiStaticPrior>(state, "wnoa_quasi_static.Prior"), base_start_point, Q0, Q2, options); };
            };
            auto prior = py::classh<QuasiStaticPrior, PyStatefulModel<QuasiStaticPrior>, StatefulModel>(
                m, "Prior",
                "The quasi-static prior: WNOA in arclength only, every node row an independent shape. Use it with "
                "EstimatorOptions.quasi_static, which places a row at each measurement time.")
                .def(py::init(make(std::type_identity<QuasiStaticPrior>()), make(std::type_identity<PyStatefulModel<QuasiStaticPrior>>())),
                     py::arg("state"), py::arg("base_start_point"), py::arg("Q0") = Matrix2N::Identity().eval(),
                     py::arg("Q2") = MatrixN::Identity().eval(), py::arg("options") = QS::PriorOptions(),
                     "`state` is a wnoa_quasi_static.ContinuumRobotState (else ValueError). base_start_point: the unary factors' base-node "
                     "pose T_ib and strain. Q0: unary covariance over [pose, strain] (only its diagonal blocks are used). "
                     "Q2: arclength-axis (pose, strain) PSD.")
                .def("options", &QuasiStaticPrior::options, py::return_value_policy::reference_internal, "The prior's options.");
            bindGridPrior<QuasiStaticPrior>(m, prior);
        }
    } // namespace

    void bindPriors(py::module_ &m)
    {
        auto wnoa = m.attr("wnoa").cast<py::module_>();
        auto wnov = m.attr("wnov").cast<py::module_>();
        auto quasi_static = m.attr("wnoa_quasi_static").cast<py::module_>();
        bindWnoa(wnoa);
        bindWnov(wnov);
        bindQuasiStatic(quasi_static);
    }
} // namespace SpacetimePython
