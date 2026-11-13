/**
 * @file bind_types.cpp
 * @brief spacetime.types: the library's base classes, each subclassable in Python.
 */

#include <typeindex>

#include <spacetime/estimator/PosteriorCovariance.hpp>

#include "common.hpp"

namespace SpacetimePython
{
    using namespace Spacetime::Types;

    namespace
    {
        /// Inserts `value` for `node`'s variable `name` into `values`, as the C++ type it was registered with.
        void insertByRegisteredType(const NodeBase &node, const std::string &name, const py::object &value, gtsam::Values &values)
        {
            const std::type_index type = node.type(name);
            const auto as = [&](auto *tag)
            {
                using T = std::remove_pointer_t<decltype(tag)>;
                T typed;
                try
                {
                    typed = value.cast<T>();
                }
                catch (const py::cast_error &)
                {
                    throw std::runtime_error("Node::insert type mismatch for variable '" + name + "'");
                }
                node.insert<T>(name, typed, values);
            };
            if (type == typeid(gtsam::Pose3))
                return as(static_cast<gtsam::Pose3 *>(nullptr));
            if (type == typeid(gtsam::Rot3))
                return as(static_cast<gtsam::Rot3 *>(nullptr));
            if (type == typeid(gtsam::Vector))
                return as(static_cast<gtsam::Vector *>(nullptr));
            if (type == typeid(Vector6))
                return as(static_cast<Vector6 *>(nullptr));
            if (type == typeid(gtsam::Vector3))
                return as(static_cast<gtsam::Vector3 *>(nullptr));
            throw py::type_error("Node::insert: variable '" + name + "' has a type the Python bindings do not support.");
        }
    } // namespace

    void bindTypes(py::module_ &m)
    {
        py::classh<KeyManager>(m, "KeyManager", "Allocates per-character gtsam.Symbol keys and records their timestamps.")
            .def(py::init<>())
            .def("nextKey", [](KeyManager &self, char chr, double timestamp) { return self.nextKey(static_cast<unsigned char>(chr), timestamp); },
                 py::arg("chr"), py::arg("timestamp"))
            .def("drainPending", &KeyManager::drainPending)
            .def("timestampOf", &KeyManager::timestampOf, py::arg("key"));

        py::classh<MeasurementBase>(m, "MeasurementBase", "A measurement at (arclength, time): its value, covariance and component mask.")
            .def(py::init<>())
            .def(py::init<double, double, const Eigen::MatrixXd &, const Eigen::MatrixXd &>(), py::arg("arclength"), py::arg("time"),
                 py::arg("value"), py::arg("covariance"))
            .def(py::init<double, double, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXi &>(), py::arg("arclength"),
                 py::arg("time"), py::arg("value"), py::arg("covariance"), py::arg("mask"))
            .def_readwrite("value", &MeasurementBase::value)
            .def_readwrite("covariance", &MeasurementBase::covariance)
            .def_readwrite("mask", &MeasurementBase::mask)
            .def_readwrite("arclength", &MeasurementBase::arclength)
            .def_readwrite("time", &MeasurementBase::time)
            .def("print", &MeasurementBase::print);

        py::classh<MeasurementQueue>(m, "MeasurementQueue", "A thread-safe queue of measurements, oldest first.")
            .def(py::init<>())
            .def("addMeasurement", &MeasurementQueue::addMeasurement<MeasurementBase>, py::arg("measurement"))
            .def("addMeasurementCallback", &MeasurementQueue::addMeasurementCallback, py::arg("callback"),
                 "Calls `callback` with every measurement added from now on; it must not call back into this queue.")
            .def("drainUpTo", &MeasurementQueue::drainUpTo, py::arg("time"))
            .def("takeAll", &MeasurementQueue::takeAll)
            .def("peekAll", &MeasurementQueue::peekAll)
            .def("size", &MeasurementQueue::size)
            .def("empty", &MeasurementQueue::empty)
            .def("lastTime", &MeasurementQueue::lastTime)
            .def("frontTime", &MeasurementQueue::frontTime);

        py::classh<Model, PyModel<Model>>(m, "Model", "A source of variables and factors for the estimator; override any of its methods.")
            .def(py::init<>())
            .def("extendState", &Model::extendState, py::arg("km"), py::arg("time"), py::arg("current_values"), py::arg("new_values"))
            .def("getNewFactors", &Model::getNewFactors, py::arg("time"), py::arg("new_values"))
            .def("onMarginalization", &Model::onMarginalization, py::arg("cutoff"), py::arg("values"))
            .def("lockedValues", &Model::lockedValues, py::return_value_policy::reference_internal)
            .def("covarianceKeyGroups", &Model::covarianceKeyGroups, py::arg("values"));

        py::classh<NodeValues, PyNodeValues>(m, "NodeValues", "The values of a node's variables, one matrix per component.")
            .def(py::init<>())
            .def_readwrite("values", &NodeValues::values, "A copy: assign the whole list to change it.")
            .def("setRandom", &NodeValues::setRandom);

        py::classh<NodeBase, PyNodeBase>(m, "NodeBase", "A node of the state at (arclength, time), holding named variables.")
            .def(py::init<KeyManager &, double, double>(), py::arg("km"), py::arg("arclength"), py::arg("time"))
            .def_readonly("arclength", &NodeBase::arclength)
            .def_readonly("time", &NodeBase::time)
            .def("keys", &NodeBase::keys)
            .def("names", &NodeBase::names)
            .def("key", &NodeBase::key, py::arg("name"))
            .def(
                "registerVariable",
                [](NodeBase &self, KeyManager &km, const std::string &name, char chr, const py::object &kind)
                {
                    auto *node = dynamic_cast<PyNodeBase *>(&self);
                    if (!node)
                        throw py::type_error("registerVariable is for the constructors of Python subclasses of NodeBase.");
                    return registerByKind(kind, [&](auto *tag)
                                          { return node->registerVariable<std::remove_pointer_t<decltype(tag)>>(km, name, static_cast<unsigned char>(chr)); });
                },
                py::arg("km"), py::arg("name"), py::arg("chr"), py::arg("kind"),
                "Adds the variable `name`, of type `kind` (gtsam.Pose3, gtsam.Rot3 or numpy.ndarray), with a new key; returns the key.")
            .def("insert", &insertByRegisteredType, py::arg("name"), py::arg("value"), py::arg("values"),
                 "Inserts `value` for the variable `name` into `values`, as the type it was registered with.")
            .def("interpolatedKeys", &NodeBase::interpolatedKeys)
            .def("getValuesBase", &NodeBase::getValuesBase, py::arg("values"))
            .def("print", py::overload_cast<const std::string &>(&NodeBase::print, py::const_), py::arg("s") = "");

        py::classh<State, PyState>(m, "State", "A robot's estimated state: its variables and nodes, extended by one stateful model.")
            .def(py::init<>())
            .def("addStateAtTime", &State::addStateAtTime, py::arg("km"), py::arg("time"))
            .def("getNearestNode", &State::getNearestNode, py::arg("arclength"), py::arg("time"))
            .def("times", &State::times, "The times of the state's node rows, ascending.")
            .def("lockedValues", &State::lockedValues, py::return_value_policy::reference_internal);

        py::classh<StatefulModel, PyStatefulModel<StatefulModel, true>, Model>(
            m, "StatefulModel",
            "A model that owns a state (e.g. a prior): extends it in time (override extendState) and "
            "resolves queries and measurements onto its nodes (by default, the nearest node).")
            .def(py::init<std::shared_ptr<State>>(), py::arg("state"))
            .def("state", &StatefulModel::state)
            .def("getOuterFactor", &StatefulModel::getOuterFactor, py::arg("time"), py::arg("arclength"), py::arg("inner_factor"))
            .def(
                "query", [](const StatefulModel &self, double time, double arclength, const gtsam::Values &values)
                { return self.query(time, arclength, values); },
                py::arg("time"), py::arg("arclength"), py::arg("values"),
                "The node values at (time, arclength), or None outside the estimated extent.")
            .def(
                "queryCovariance",
                [](const StatefulModel &self, double time, double arclength, const gtsam::Values &values,
                   std::shared_ptr<Spacetime::PosteriorCovariance> posterior)
                {
                    Eigen::MatrixXd conditional, posterior_covariance;
                    auto node = self.query(time, arclength, values, &conditional, posterior.get(), &posterior_covariance);
                    return py::make_tuple(node, conditional, posterior_covariance);
                },
                py::arg("time"), py::arg("arclength"), py::arg("values"), py::arg("posterior") = nullptr,
                "(node values, conditional covariance, posterior covariance) at (time, arclength); "
                "the posterior covariance is empty without `posterior`.");

        py::classh<SensorBase, PySensor<SensorBase, true>, Model, MeasurementQueue>(
            m, "SensorBase", "A sensor: turns each queued measurement into a factor (override getFactor) on its stateful model's interpolated keys.")
            .def(py::init<pStateful>(), py::arg("stateful"))
            .def("getFactor", &SensorBase::getFactor, py::arg("measurement"))
            .def("stateful", &SensorBase::stateful)
            .def_readwrite("data_rows", &SensorBase::data_rows)
            .def_readwrite("data_cols", &SensorBase::data_cols);

        py::classh<ActuatorBase, PyActuator, StatefulModel, MeasurementQueue>(
            m, "ActuatorBase", "An input acting along the robot: each measured input becomes a variable shared by its factors (override getFactor).")
            .def(py::init([](pStateful stateful, std::vector<double> arclengths, unsigned int input_dim, char input_symbol)
                          { return new PyActuator(std::move(stateful), std::move(arclengths), input_dim, static_cast<unsigned char>(input_symbol)); }),
                 py::arg("stateful"), py::arg("arclengths"), py::arg("input_dim"), py::arg("input_symbol") = 'u')
            .def("getFactor", &ActuatorBase::getFactor, py::arg("measurement"), py::arg("input_key"), py::arg("arclength"))
            .def("stateful", &ActuatorBase::stateful)
            .def("arclengths", &ActuatorBase::arclengths)
            .def("inputDim", &ActuatorBase::inputDim)
            .def("inputKeys", &ActuatorBase::inputKeys);

        py::classh<Robot>(m, "Robot", "A convenience list of the models making up one robot (e.g. its prior and sensors), for Estimator.addRobot.")
            .def(py::init<>())
            .def("addModel", &Robot::addModel, py::arg("model"))
            .def("models", &Robot::models);
    }
} // namespace SpacetimePython
