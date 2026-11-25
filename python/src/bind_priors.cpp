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

        using Matrix6 = Eigen::Matrix<double, 6, 6>;
        using Override6 = Spacetime::Priors::CovarianceOverride<Matrix6>;
        using Profile6 = Spacetime::Priors::CovarianceProfile<Matrix6>;

        constexpr const char *kOverrideKeyDoc =
            " Time-axis covariances are keyed by a node's arclength, arclength-axis ones by their segment's lower node "
            "(-1: the tip, or the last segment); keys must be node arclengths of the prior's state (within 1e-8).";

        /// priors.CovarianceAxis, priors.CovarianceOverride and priors.CovarianceProfile (6 x 6, all the priors' Q's on Pose3).
        void bindCovarianceProfiles(py::module_ &m)
        {
            using Axis = Spacetime::Priors::CovarianceAxis;
            py::enum_<Axis>(m, "CovarianceAxis", "Which grid edges a prior covariance belongs to, which fixes how its overrides are keyed.")
                .value("Time", Axis::Time, "One factor per node (between consecutive rows), keyed by the node's arclength.")
                .value("Space", Axis::Space, "One factor per segment [s_n, s_n+1], keyed by its lower arclength s_n.");

            py::class_<Override6>(m, "CovarianceOverride",
                                  (std::string("Arclength-dependent overrides of one prior covariance, applied at every time step. Each node (or segment) "
                                               "resolves as: the nominal value, then the function if set, then every set/scale call at its key in call "
                                               "order (a later one wins).") +
                                   kOverrideKeyDoc)
                                      .c_str())
                .def(py::init<>())
                .def("set", py::overload_cast<double, const Matrix6 &>(&Override6::set), py::arg("arclength"), py::arg("value"),
                     py::return_value_policy::reference_internal, "Replace the covariance at node (or segment) `arclength` by `value`. Returns self.")
                .def("set", py::overload_cast<Override6::Function>(&Override6::set), py::arg("function"), py::return_value_policy::reference_internal,
                     "Set every node (or segment) to function(key arclength, nominal), evaluated when the prior is constructed, before the "
                     "point entries. Returns self.")
                .def("scale", &Override6::scale, py::arg("arclength"), py::arg("factor"), py::return_value_policy::reference_internal,
                     "Multiply the covariance resolved so far at node (or segment) `arclength` by `factor`. Returns self.")
                .def("empty", &Override6::empty, "True if nothing overrides the nominal value.");

            py::class_<Profile6>(m, "CovarianceProfile", "One prior covariance resolved over the robot's nodes (Time) or segments (Space).")
                .def("nominal", &Profile6::nominal, "The value wherever nothing overrides it.")
                .def("axis", &Profile6::axis, "Whether the covariance is per node (Time) or per segment (Space).")
                .def("isConstant", &Profile6::isConstant, "True if no overrides were given.")
                .def("size", &Profile6::size, "Number of nodes (Time) or segments (Space) resolved; 0 when constant.")
                .def("at", &Profile6::at, py::arg("index"), "The value at node (Time) or segment (Space) `index`; the nominal one when constant.")
                .def("atArclength", &Profile6::atArclength, py::arg("arclength"),
                     "The value of the grid cell anchored at `arclength` (the last node at or before it; for Space clamped to the last "
                     "segment); None gives the nominal value.");
        }

        constexpr const char *kWnoaDefaultCovariancesDoc =
            "The default tuned (Q0, Q1, Q2, Q3, Q4), in Prior's argument order: the experiment tuning of the paper "
            "(papers/.../configs/experiment_base.json, times its prior_factor 0.5), with Q0's rotational strain permuted from "
            "the paper's z tangent axis to the library's x.";
        constexpr const char *kWnovDefaultCovariancesDoc = "The default (Q0, Q1, Q2), in Prior's argument order: identity.";
        constexpr const char *kQuasiStaticDefaultCovariancesDoc =
            "The default tuned (Q0, Q2), in Prior's argument order: wnoa.DefaultCovariances()'s [pose, strain] block of Q0, and its Q2.";

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

            py::classh<WnoaValues, Spacetime::Types::NodeValuesBase>(m, "NodeValues", "A WNOA node's pose, strain (epsilon) and velocity (varpi), [rot, trans].")
                .def(py::init<>(), "Identity pose, straight strain [0, 0, 0, 1, 0, 0] (the backbone along body x) and zero velocity.")
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
                .def_static("Pose", &pointOptional<gtsam::Key, &Keys::Pose, &Keys::Pose>, py::arg("point") = py::none(), "The interpolated pose key. `point` selects the key at the point-th arclength of a multi-point factor (see types.InterpolatedKeyAt).")
                .def_static("Strain", &pointOptional<gtsam::Key, &Keys::Strain, &Keys::Strain>, py::arg("point") = py::none(), "The interpolated strain key. `point` selects the key at the point-th arclength of a multi-point factor (see types.InterpolatedKeyAt).")
                .def_static("Velocity", &pointOptional<gtsam::Key, &Keys::Velocity, &Keys::Velocity>, py::arg("point") = py::none(), "The interpolated velocity key. `point` selects the key at the point-th arclength of a multi-point factor (see types.InterpolatedKeyAt).")
                .def_static("keys", &pointOptional<gtsam::KeyVector, &Keys::keys, &Keys::keys>, py::arg("point") = py::none(), "Every interpolated key of the node type. `point` selects the key at the point-th arclength of a multi-point factor (see types.InterpolatedKeyAt).");

            bindContinuumRobotState<WnoaNode>(m, "ContinuumRobotState");

            using Overrides = WNOA::CovarianceOverrides<Pose>;
            using Profiles = WNOA::CovarianceProfiles<Pose>;
            py::class_<Overrides>(m, "CovarianceOverrides",
                                  (std::string("Per-arclength values of the WNOA prior's Q1..Q4 (each a priors.CovarianceOverride), given to Prior "
                                               "at construction. Q1, Q3: time axis (per node). Q2, Q4: arclength axis (per segment).") +
                                   kOverrideKeyDoc)
                                      .c_str())
                .def(py::init<>())
                .def_readwrite("Q1", &Overrides::Q1, "Time-axis (pose, velocity) PSD overrides, per node.")
                .def_readwrite("Q2", &Overrides::Q2, "Arclength-axis (pose, strain) PSD overrides, per segment.")
                .def_readwrite("Q3", &Overrides::Q3, "Strain random-walk-in-time PSD overrides, per node.")
                .def_readwrite("Q4", &Overrides::Q4, "Velocity random-walk-in-arclength PSD overrides, per segment.")
                .def("overrideQ1", py::overload_cast<double, const MatrixN &>(&Overrides::overrideQ1), py::arg("arclength"), py::arg("Q"),
                     py::return_value_policy::reference_internal, "Q1 at the node at `arclength`. Returns self.")
                .def("overrideQ2", py::overload_cast<double, const MatrixN &>(&Overrides::overrideQ2), py::arg("arclength"), py::arg("Q"),
                     py::return_value_policy::reference_internal, "Q2 on the segment starting at the node at `arclength`. Returns self.")
                .def("overrideQ3", py::overload_cast<double, const MatrixN &>(&Overrides::overrideQ3), py::arg("arclength"), py::arg("Q"),
                     py::return_value_policy::reference_internal, "Q3 at the node at `arclength`. Returns self.")
                .def("overrideQ4", py::overload_cast<double, const MatrixN &>(&Overrides::overrideQ4), py::arg("arclength"), py::arg("Q"),
                     py::return_value_policy::reference_internal, "Q4 on the segment starting at the node at `arclength`. Returns self.")
                .def("overrideQ1", py::overload_cast<Override6::Function>(&Overrides::overrideQ1), py::arg("function"),
                     py::return_value_policy::reference_internal, "Q1 at every node: function(arclength, nominal Q1). Returns self.")
                .def("overrideQ2", py::overload_cast<Override6::Function>(&Overrides::overrideQ2), py::arg("function"),
                     py::return_value_policy::reference_internal, "Q2 on every segment: function(lower arclength, nominal Q2). Returns self.")
                .def("overrideQ3", py::overload_cast<Override6::Function>(&Overrides::overrideQ3), py::arg("function"),
                     py::return_value_policy::reference_internal, "Q3 at every node: function(arclength, nominal Q3). Returns self.")
                .def("overrideQ4", py::overload_cast<Override6::Function>(&Overrides::overrideQ4), py::arg("function"),
                     py::return_value_policy::reference_internal, "Q4 on every segment: function(lower arclength, nominal Q4). Returns self.")
                .def("empty", &Overrides::empty, "True if nothing is overridden.");
            py::class_<Profiles>(m, "CovarianceProfiles", "The WNOA prior's Q1..Q4 resolved over its state (each a priors.CovarianceProfile).")
                .def_readonly("Q1", &Profiles::Q1, "Time-axis (pose, velocity) PSD, per node.")
                .def_readonly("Q2", &Profiles::Q2, "Arclength-axis (pose, strain) PSD, per segment.")
                .def_readonly("Q3", &Profiles::Q3, "Strain random-walk-in-time PSD, per node.")
                .def_readonly("Q4", &Profiles::Q4, "Velocity random-walk-in-arclength PSD, per segment.");

            const auto make = [](auto tag)
            {
                using T = typename decltype(tag)::type;
                return [](std::shared_ptr<Spacetime::Types::State> state, const WnoaValues &base_start_point, const Matrix3N &Q0,
                          const MatrixN &Q1, const MatrixN &Q2, const MatrixN &Q3, const MatrixN &Q4, const WNOA::PriorOptions &options,
                          const Overrides &overrides)
                { return new T(gridState<WnoaPrior>(state, "wnoa.Prior"), base_start_point, Q0, Q1, Q2, Q3, Q4, options, overrides); };
            };
            const auto makeDefault = [](auto tag)
            {
                using T = typename decltype(tag)::type;
                return [](std::shared_ptr<Spacetime::Types::State> state, const WnoaValues &base_start_point, const WNOA::PriorOptions &options,
                          const Overrides &overrides)
                { return new T(gridState<WnoaPrior>(state, "wnoa.Prior"), base_start_point, options, overrides); };
            };
            auto prior = py::classh<WnoaPrior, PyStatefulModel<WnoaPrior>, StatefulModel>(m, "Prior",
                                                                            "The white-noise-on-acceleration GP prior over a robot's grid state.")
                .def(py::init(make(std::type_identity<WnoaPrior>()), make(std::type_identity<PyStatefulModel<WnoaPrior>>())), py::arg("state"),
                     py::arg("base_start_point"), py::arg("Q0"), py::arg("Q1"), py::arg("Q2"), py::arg("Q3"), py::arg("Q4"),
                     py::arg("options") = WNOA::PriorOptions(), py::arg("overrides") = Overrides(),
                     "`state` is a wnoa.ContinuumRobotState (else ValueError). base_start_point: the unary factor's base-node values "
                     "(pose T_ib, strain and velocity [rot, trans]). Q0: unary covariance over [pose, strain, velocity]. "
                     "Q1: time-axis (pose, velocity) PSD. Q2: arclength-axis (pose, strain) PSD. Q3: strain random walk in time PSD. "
                     "Q4: velocity random walk in arclength PSD. Q1..Q4 are taken as diagonal. overrides: per-arclength Q1..Q4 "
                     "(wnoa.CovarianceOverrides); raises ValueError for a key that is not a node arclength.")
                .def(py::init(makeDefault(std::type_identity<WnoaPrior>()), makeDefault(std::type_identity<PyStatefulModel<WnoaPrior>>())),
                     py::arg("state"), py::arg("base_start_point"), py::arg("options") = WNOA::PriorOptions(), py::arg("overrides") = Overrides(),
                     "A prior with the default tuned covariances (wnoa.DefaultCovariances()), and optional per-arclength overrides.")
                .def("options", &WnoaPrior::options, py::return_value_policy::reference_internal, "The prior's options.")
                .def("covarianceProfiles", &WnoaPrior::covarianceProfiles, py::return_value_policy::reference_internal,
                     "Q1..Q4 resolved over the state's nodes and segments (wnoa.CovarianceProfiles).")
                .def_static("DefaultCovariances", &WnoaPrior::DefaultCovariances, kWnoaDefaultCovariancesDoc);
            bindGridPrior<WnoaPrior>(m, prior);
            m.def("DefaultCovariances", &WnoaPrior::DefaultCovariances, kWnoaDefaultCovariancesDoc);
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

            py::classh<WnovValues, Spacetime::Types::NodeValuesBase>(m, "NodeValues", "A WNOV node's pose.")
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
                .def_static("Pose", &pointOptional<gtsam::Key, &Keys::Pose, &Keys::Pose>, py::arg("point") = py::none(), "The interpolated pose key. `point` selects the key at the point-th arclength of a multi-point factor (see types.InterpolatedKeyAt).")
                .def_static("keys", &pointOptional<gtsam::KeyVector, &Keys::keys, &Keys::keys>, py::arg("point") = py::none(), "Every interpolated key of the node type. `point` selects the key at the point-th arclength of a multi-point factor (see types.InterpolatedKeyAt).");

            bindContinuumRobotState<WnovNode>(m, "ContinuumRobotState");

            using Overrides = WNOV::CovarianceOverrides<Pose>;
            using Profiles = WNOV::CovarianceProfiles<Pose>;
            py::class_<Overrides>(m, "CovarianceOverrides",
                                  (std::string("Per-arclength values of the WNOV prior's Q1 and Q2 (each a priors.CovarianceOverride), given to "
                                               "Prior at construction. Q1: arclength axis (per segment). Q2: time axis (per node).") +
                                   kOverrideKeyDoc)
                                      .c_str())
                .def(py::init<>())
                .def_readwrite("Q1", &Overrides::Q1, "Arclength random-walk PSD overrides, per segment.")
                .def_readwrite("Q2", &Overrides::Q2, "Time random-walk PSD overrides, per node.")
                .def("overrideQ1", py::overload_cast<double, const MatrixN &>(&Overrides::overrideQ1), py::arg("arclength"), py::arg("Q"),
                     py::return_value_policy::reference_internal, "Q1 on the segment starting at the node at `arclength`. Returns self.")
                .def("overrideQ2", py::overload_cast<double, const MatrixN &>(&Overrides::overrideQ2), py::arg("arclength"), py::arg("Q"),
                     py::return_value_policy::reference_internal, "Q2 at the node at `arclength`. Returns self.")
                .def("overrideQ1", py::overload_cast<Override6::Function>(&Overrides::overrideQ1), py::arg("function"),
                     py::return_value_policy::reference_internal, "Q1 on every segment: function(lower arclength, nominal Q1). Returns self.")
                .def("overrideQ2", py::overload_cast<Override6::Function>(&Overrides::overrideQ2), py::arg("function"),
                     py::return_value_policy::reference_internal, "Q2 at every node: function(arclength, nominal Q2). Returns self.")
                .def("empty", &Overrides::empty, "True if nothing is overridden.");
            py::class_<Profiles>(m, "CovarianceProfiles", "The WNOV prior's Q1 and Q2 resolved over its state (each a priors.CovarianceProfile).")
                .def_readonly("Q1", &Profiles::Q1, "Arclength random-walk PSD, per segment.")
                .def_readonly("Q2", &Profiles::Q2, "Time random-walk PSD, per node.");

            const auto make = [](auto tag)
            {
                using T = typename decltype(tag)::type;
                return [](std::shared_ptr<Spacetime::Types::State> state, const Pose &T0, const Vector6 &epsilon0, const MatrixN &Q0,
                          const MatrixN &Q1, const MatrixN &Q2, const WNOV::PriorOptions &options, const Overrides &overrides)
                { return new T(gridState<WnovPrior>(state, "wnov.Prior"), T0, epsilon0, Q0, Q1, Q2, options, overrides); };
            };
            const auto makeDefault = [](auto tag)
            {
                using T = typename decltype(tag)::type;
                return [](std::shared_ptr<Spacetime::Types::State> state, const Pose &T0, const Vector6 &epsilon0, const WNOV::PriorOptions &options,
                          const Overrides &overrides)
                { return new T(gridState<WnovPrior>(state, "wnov.Prior"), T0, epsilon0, options, overrides); };
            };
            auto prior = py::classh<WnovPrior, PyStatefulModel<WnovPrior>, StatefulModel>(m, "Prior",
                                                                            "The white-noise-on-velocity GP prior over a robot's grid state.")
                .def(py::init(make(std::type_identity<WnovPrior>()), make(std::type_identity<PyStatefulModel<WnovPrior>>())), py::arg("state"), py::arg("T0"),
                     py::arg("epsilon0"), py::arg("Q0"), py::arg("Q1"), py::arg("Q2"), py::arg("options") = WNOV::PriorOptions(),
                     py::arg("overrides") = Overrides(),
                     "`state` is a wnov.ContinuumRobotState (else ValueError). T0: the base pose T_ib the unary factor holds each row's base node to. "
                     "epsilon0: nominal strain [rot, trans]. Q0: unary covariance. Q1: arclength random walk PSD (per metre). "
                     "Q2: time random walk PSD (per second). overrides: per-arclength Q1 and Q2 (wnov.CovarianceOverrides); raises "
                     "ValueError for a key that is not a node arclength.")
                .def(py::init(makeDefault(std::type_identity<WnovPrior>()), makeDefault(std::type_identity<PyStatefulModel<WnovPrior>>())),
                     py::arg("state"), py::arg("T0"), py::arg("epsilon0"), py::arg("options") = WNOV::PriorOptions(), py::arg("overrides") = Overrides(),
                     "A prior with the default covariances (wnov.DefaultCovariances()), and optional per-arclength overrides.")
                .def("options", &WnovPrior::options, py::return_value_policy::reference_internal, "The prior's options.")
                .def("covarianceProfiles", &WnovPrior::covarianceProfiles, py::return_value_policy::reference_internal,
                     "Q1 and Q2 resolved over the state's segments and nodes (wnov.CovarianceProfiles).")
                .def_static("DefaultCovariances", &WnovPrior::DefaultCovariances, kWnovDefaultCovariancesDoc);
            bindGridPrior<WnovPrior>(m, prior);
            m.def("DefaultCovariances", &WnovPrior::DefaultCovariances, kWnovDefaultCovariancesDoc);
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

            py::classh<QuasiStaticValues, Spacetime::Types::NodeValuesBase>(m, "NodeValues", "A quasi-static node's pose and strain (epsilon), [rot, trans].")
                .def(py::init<>(), "Identity pose and straight strain [0, 0, 0, 1, 0, 0] (the backbone along body x).")
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
                .def_static("Pose", &pointOptional<gtsam::Key, &Keys::Pose, &Keys::Pose>, py::arg("point") = py::none(), "The interpolated pose key. `point` selects the key at the point-th arclength of a multi-point factor (see types.InterpolatedKeyAt).")
                .def_static("Strain", &pointOptional<gtsam::Key, &Keys::Strain, &Keys::Strain>, py::arg("point") = py::none(), "The interpolated strain key. `point` selects the key at the point-th arclength of a multi-point factor (see types.InterpolatedKeyAt).")
                .def_static("keys", &pointOptional<gtsam::KeyVector, &Keys::keys, &Keys::keys>, py::arg("point") = py::none(), "Every interpolated key of the node type. `point` selects the key at the point-th arclength of a multi-point factor (see types.InterpolatedKeyAt).");

            bindContinuumRobotState<QuasiStaticNode>(m, "ContinuumRobotState");

            using Overrides = QS::CovarianceOverrides<Pose>;
            using Profiles = QS::CovarianceProfiles<Pose>;
            py::class_<Overrides>(m, "CovarianceOverrides",
                                  (std::string("Per-segment values of the quasi-static prior's Q2 (a priors.CovarianceOverride), given to Prior at "
                                               "construction.") +
                                   kOverrideKeyDoc)
                                      .c_str())
                .def(py::init<>())
                .def_readwrite("Q2", &Overrides::Q2, "Arclength-axis (pose, strain) PSD overrides, per segment.")
                .def("overrideQ2", py::overload_cast<double, const MatrixN &>(&Overrides::overrideQ2), py::arg("arclength"), py::arg("Q"),
                     py::return_value_policy::reference_internal, "Q2 on the segment starting at the node at `arclength`. Returns self.")
                .def("overrideQ2", py::overload_cast<Override6::Function>(&Overrides::overrideQ2), py::arg("function"),
                     py::return_value_policy::reference_internal, "Q2 on every segment: function(lower arclength, nominal Q2). Returns self.")
                .def("empty", &Overrides::empty, "True if nothing is overridden.");
            py::class_<Profiles>(m, "CovarianceProfiles", "The quasi-static prior's Q2 resolved over its state (a priors.CovarianceProfile).")
                .def_readonly("Q2", &Profiles::Q2, "Arclength-axis (pose, strain) PSD, per segment.");

            const auto make = [](auto tag)
            {
                using T = typename decltype(tag)::type;
                return [](std::shared_ptr<Spacetime::Types::State> state, const QuasiStaticValues &base_start_point, const Matrix2N &Q0,
                          const MatrixN &Q2, const QS::PriorOptions &options, const Overrides &overrides)
                { return new T(gridState<QuasiStaticPrior>(state, "wnoa_quasi_static.Prior"), base_start_point, Q0, Q2, options, overrides); };
            };
            const auto makeDefault = [](auto tag)
            {
                using T = typename decltype(tag)::type;
                return [](std::shared_ptr<Spacetime::Types::State> state, const QuasiStaticValues &base_start_point, const QS::PriorOptions &options,
                          const Overrides &overrides)
                { return new T(gridState<QuasiStaticPrior>(state, "wnoa_quasi_static.Prior"), base_start_point, options, overrides); };
            };
            auto prior = py::classh<QuasiStaticPrior, PyStatefulModel<QuasiStaticPrior>, StatefulModel>(
                m, "Prior",
                "The quasi-static prior: WNOA in arclength only, every node row an independent shape. Use it with "
                "EstimatorOptions.quasi_static, which places a row at each measurement time.")
                .def(py::init(make(std::type_identity<QuasiStaticPrior>()), make(std::type_identity<PyStatefulModel<QuasiStaticPrior>>())),
                     py::arg("state"), py::arg("base_start_point"), py::arg("Q0"), py::arg("Q2"), py::arg("options") = QS::PriorOptions(),
                     py::arg("overrides") = Overrides(),
                     "`state` is a wnoa_quasi_static.ContinuumRobotState (else ValueError). base_start_point: the unary factors' base-node "
                     "pose T_ib and strain. Q0: unary covariance over [pose, strain] (only its diagonal blocks are used). "
                     "Q2: arclength-axis (pose, strain) PSD. overrides: per-segment Q2 (wnoa_quasi_static.CovarianceOverrides); raises "
                     "ValueError for a key that is not a node arclength.")
                .def(py::init(makeDefault(std::type_identity<QuasiStaticPrior>()), makeDefault(std::type_identity<PyStatefulModel<QuasiStaticPrior>>())),
                     py::arg("state"), py::arg("base_start_point"), py::arg("options") = QS::PriorOptions(), py::arg("overrides") = Overrides(),
                     "A prior with the default tuned covariances (wnoa_quasi_static.DefaultCovariances()), and optional per-segment overrides.")
                .def("options", &QuasiStaticPrior::options, py::return_value_policy::reference_internal, "The prior's options.")
                .def("covarianceProfiles", &QuasiStaticPrior::covarianceProfiles, py::return_value_policy::reference_internal,
                     "Q2 resolved over the state's segments (wnoa_quasi_static.CovarianceProfiles).")
                .def_static("DefaultCovariances", &QuasiStaticPrior::DefaultCovariances, kQuasiStaticDefaultCovariancesDoc);
            bindGridPrior<QuasiStaticPrior>(m, prior);
            m.def("DefaultCovariances", &QuasiStaticPrior::DefaultCovariances, kQuasiStaticDefaultCovariancesDoc);
        }
    } // namespace

    void bindPriors(py::module_ &m)
    {
        auto wnoa = m.attr("wnoa").cast<py::module_>();
        auto wnov = m.attr("wnov").cast<py::module_>();
        auto quasi_static = m.attr("wnoa_quasi_static").cast<py::module_>();
        bindCovarianceProfiles(m);
        bindWnoa(wnoa);
        bindWnov(wnov);
        bindQuasiStatic(quasi_static);
    }
} // namespace SpacetimePython
