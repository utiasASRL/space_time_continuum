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
                .def("innerFactor", &Factor::innerFactor)
                .def("time", &Factor::time)
                .def("arclength", &Factor::arclength)
                .def("fixedNoiseModel", &Factor::fixedNoiseModel)
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

            py::enum_<WNOA::FactorType>(m, "FactorType")
                .value("WNOA", WNOA::FactorType::WNOA, "gtsam.WnoaMotionFactor + BetweenFactor: the GP on the Lie algebra.")
                .value("Legacy", WNOA::FactorType::Legacy, "The binary time and space factors on the group, with the legacy code's strain-velocity coupling term.")
                .value("Paper", WNOA::FactorType::Paper, "The binary time and space factors on the group, without the coupling term (TRO paper, Eq. 20).");

            py::class_<WNOA::PriorOptions>(m, "PriorOptions")
                .def(py::init<>())
                .def_readwrite("factor_type", &WNOA::PriorOptions::factor_type)
                .def_readwrite("fixed_noise_model", &WNOA::PriorOptions::fixed_noise_model)
                .def_readwrite("cache_noise_model", &WNOA::PriorOptions::cache_noise_model)
                .def_readwrite("remove_unary", &WNOA::PriorOptions::remove_unary);

            py::classh<WnoaValues, Spacetime::Types::NodeValues>(m, "Values", "A WNOA node's pose, strain (epsilon) and velocity (varpi), [rot, trans].")
                .def(py::init<>())
                .def_readonly_static("dim", &WnoaValues::dim)
                .def("pose", py::overload_cast<>(&WnoaValues::pose, py::const_))
                .def("pose", py::overload_cast<const Pose &>(&WnoaValues::pose), py::arg("value"))
                .def("epsilon", py::overload_cast<>(&WnoaValues::epsilon, py::const_))
                .def("epsilon", py::overload_cast<const Vector6 &>(&WnoaValues::epsilon), py::arg("value"))
                .def("varpi", py::overload_cast<>(&WnoaValues::varpi, py::const_))
                .def("varpi", py::overload_cast<const Vector6 &>(&WnoaValues::varpi), py::arg("value"));

            py::classh<WnoaNode, Spacetime::Types::NodeBase>(m, "Node", "A WNOA node: pose ('p'), strain ('s') and velocity ('v') variables.")
                .def(py::init<KeyManager &, double, double>(), py::arg("km"), py::arg("arclength"), py::arg("time"))
                .def("poseKey", &WnoaNode::poseKey)
                .def("strainKey", &WnoaNode::strainKey)
                .def("velocityKey", &WnoaNode::velocityKey)
                .def("getValues", &WnoaNode::getValues, py::arg("values"))
                .def("setValues", &WnoaNode::setValues, py::arg("values"), py::arg("node_values"));

            using Keys = WNOA::InterpolatedKeys<Pose>;
            py::class_<Keys>(m, "InterpolatedKeys", "The keys a sensor's factor is built on, resolved onto the prior's nodes.")
                .def_static("Pose", &Keys::Pose)
                .def_static("Strain", &Keys::Strain)
                .def_static("Velocity", &Keys::Velocity)
                .def_static("keys", &Keys::keys);

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
                     py::arg("Q4") = MatrixN::Identity().eval(), py::arg("options") = WNOA::PriorOptions())
                .def("options", &WnoaPrior::options, py::return_value_policy::reference_internal);
            bindGridPrior<WnoaPrior>(m, prior);
        }

        void bindWnov(py::module_ &m)
        {
            namespace WNOV = Spacetime::Priors::WNOV;
            using MatrixN = WnovPrior::MatrixN;

            py::class_<WNOV::PriorOptions>(m, "PriorOptions")
                .def(py::init<>())
                .def_readwrite("fixed_noise_model", &WNOV::PriorOptions::fixed_noise_model)
                .def_readwrite("cache_noise_model", &WNOV::PriorOptions::cache_noise_model)
                .def_readwrite("remove_unary", &WNOV::PriorOptions::remove_unary);

            py::classh<WnovValues, Spacetime::Types::NodeValues>(m, "Values", "A WNOV node's pose.")
                .def(py::init<>())
                .def(py::init<const Pose &>(), py::arg("value"))
                .def_readonly_static("dim", &WnovValues::dim)
                .def("pose", py::overload_cast<>(&WnovValues::pose, py::const_))
                .def("pose", py::overload_cast<const Pose &>(&WnovValues::pose), py::arg("value"));

            py::classh<WnovNode, Spacetime::Types::NodeBase>(m, "Node", "A WNOV node: a pose ('p') variable.")
                .def(py::init<KeyManager &, double, double>(), py::arg("km"), py::arg("arclength"), py::arg("time"))
                .def("poseKey", &WnovNode::poseKey)
                .def("getValues", &WnovNode::getValues, py::arg("values"))
                .def("setValues", &WnovNode::setValues, py::arg("values"), py::arg("node_values"));

            using Keys = WNOV::InterpolatedKeys<Pose>;
            py::class_<Keys>(m, "InterpolatedKeys", "The keys a sensor's factor is built on, resolved onto the prior's nodes.")
                .def_static("Pose", &Keys::Pose)
                .def_static("keys", &Keys::keys);

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
                     py::arg("Q1") = MatrixN::Identity().eval(), py::arg("Q2") = MatrixN::Identity().eval(), py::arg("options") = WNOV::PriorOptions())
                .def("options", &WnovPrior::options, py::return_value_policy::reference_internal);
            bindGridPrior<WnovPrior>(m, prior);
        }

        void bindQuasiStatic(py::module_ &m)
        {
            namespace QS = Spacetime::Priors::WNOAQuasiStatic;
            using MatrixN = QuasiStaticPrior::MatrixN;
            using Matrix2N = QuasiStaticPrior::Matrix2N;

            py::class_<QS::PriorOptions>(m, "PriorOptions")
                .def(py::init<>())
                .def_readwrite("fixed_noise_model", &QS::PriorOptions::fixed_noise_model)
                .def_readwrite("cache_noise_model", &QS::PriorOptions::cache_noise_model);

            py::classh<QuasiStaticValues, Spacetime::Types::NodeValues>(m, "Values", "A quasi-static node's pose and strain (epsilon), [rot, trans].")
                .def(py::init<>())
                .def(py::init<const Pose &, const Vector6 &>(), py::arg("pose"), py::arg("epsilon"))
                .def_readonly_static("dim", &QuasiStaticValues::dim)
                .def("pose", py::overload_cast<>(&QuasiStaticValues::pose, py::const_))
                .def("pose", py::overload_cast<const Pose &>(&QuasiStaticValues::pose), py::arg("value"))
                .def("epsilon", py::overload_cast<>(&QuasiStaticValues::epsilon, py::const_))
                .def("epsilon", py::overload_cast<const Vector6 &>(&QuasiStaticValues::epsilon), py::arg("value"));

            py::classh<QuasiStaticNode, Spacetime::Types::NodeBase>(m, "Node", "A quasi-static node: pose ('p') and strain ('s') variables.")
                .def(py::init<KeyManager &, double, double>(), py::arg("km"), py::arg("arclength"), py::arg("time"))
                .def("poseKey", &QuasiStaticNode::poseKey)
                .def("strainKey", &QuasiStaticNode::strainKey)
                .def("getValues", &QuasiStaticNode::getValues, py::arg("values"))
                .def("setValues", &QuasiStaticNode::setValues, py::arg("values"), py::arg("node_values"));

            using Keys = QS::InterpolatedKeys<Pose>;
            py::class_<Keys>(m, "InterpolatedKeys", "The keys a sensor's factor is built on, resolved onto the prior's nodes.")
                .def_static("Pose", &Keys::Pose)
                .def_static("Strain", &Keys::Strain)
                .def_static("keys", &Keys::keys);

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
                     py::arg("Q2") = MatrixN::Identity().eval(), py::arg("options") = QS::PriorOptions())
                .def("options", &QuasiStaticPrior::options, py::return_value_policy::reference_internal);
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
