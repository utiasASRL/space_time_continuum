/**
 * @file bind_types.cpp
 * @brief spacetime.types: the library's base classes, each subclassable in Python.
 */

#include <optional>
#include <string>
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

    namespace
    {
        /// A Python number as a 1 x 1 matrix, or anything else cast to a matrix. @throws py::type_error otherwise.
        Eigen::MatrixXd scalarOrMatrix(const py::object &object, const char *name)
        {
            if (py::isinstance<py::float_>(object) || py::isinstance<py::int_>(object))
                return Eigen::MatrixXd::Constant(1, 1, object.cast<double>());
            try
            {
                return object.cast<Eigen::MatrixXd>();
            }
            catch (const py::cast_error &)
            {
                throw py::type_error(std::string("MeasurementBase: `") + name + "` must be a number or a matrix.");
            }
        }
    } // namespace

    void bindTypes(py::module_ &m)
    {
        m.def("InterpolatedKeyAt", &Spacetime::Types::InterpolatedKeyAt, py::arg("key"), py::arg("point"),
              "Interpolated key `key` (e.g. wnoa.InterpolatedKeys.Pose()) at the point-th arclength of a multi-point factor "
              "(see ActuatorBase.getFactors); the node types' accessors take it as an argument, e.g. Pose(point). "
              "Raises ValueError for an already point-indexed key.");
        m.def("isPointKey", &Spacetime::Types::isPointKey, py::arg("key"), "Whether `key` is a point-indexed interpolated key.");
        m.def("pointOf", &Spacetime::Types::pointOf, py::arg("key"), "The point index of a point-indexed key; None for any other key.");
        m.def("placeholderOf", &Spacetime::Types::placeholderOf, py::arg("key"), "The plain interpolated key a point-indexed key stands for; any other key unchanged.");

        py::classh<KeyManager>(m, "KeyManager", "Allocates per-character gtsam.Symbol keys and records their timestamps.")
            .def(py::init<>())
            .def("nextKey", [](KeyManager &self, char chr, double timestamp) { return self.nextKey(static_cast<unsigned char>(chr), timestamp); },
                 py::arg("chr"), py::arg("timestamp"), "A new key Symbol(chr, n), n counting up per character, stamped with `timestamp` [s].")
            .def("drainPending", &KeyManager::drainPending,
                 "{key: timestamp} of the keys made since the last call, emptying that set (the estimator calls it once per estimate()).")
            .def("timestampOf", &KeyManager::timestampOf, py::arg("key"), "The timestamp `key` was made with; raises IndexError for a key not made here.");

        py::classh<MeasurementBase>(m, "MeasurementBase", "A measurement at (arclength, time): its value, covariance and component mask.")
            .def(py::init<>())
            .def(py::init<double, double, const Eigen::MatrixXd &, const Eigen::MatrixXd &>(), py::arg("arclength"), py::arg("time"),
                 py::arg("value"), py::arg("covariance"), "Every component of `value` marked valid.")
            .def(py::init<double, double, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXi &>(), py::arg("arclength"),
                 py::arg("time"), py::arg("value"), py::arg("covariance"), py::arg("mask"), "`mask` has the shape of `value`: 1 valid, 0 ignored.")
            .def(py::init(
                     [](double time, const py::object &value, const py::object &covariance, double arclength, const std::optional<Eigen::MatrixXi> &mask)
                     {
                         const Eigen::MatrixXd v = scalarOrMatrix(value, "value"), c = scalarOrMatrix(covariance, "covariance");
                         return mask ? std::make_shared<MeasurementBase>(arclength, time, v, c, *mask) : std::make_shared<MeasurementBase>(arclength, time, v, c);
                     }),
                 py::kw_only(), py::arg("time"), py::arg("value"), py::arg("covariance"), py::arg("arclength") = 0.0, py::arg("mask") = py::none(),
                 "Keyword-only: `value` and `covariance` may be plain numbers (a 1 x 1 measurement, e.g. a tendon length), and "
                 "`arclength` defaults to 0 (actuators ignore it).")
            .def_readwrite("value", &MeasurementBase::value, "The measured value; its size and meaning depend on the sensor.")
            .def_readwrite("covariance", &MeasurementBase::covariance, "The value's covariance; its size and meaning depend on the sensor.")
            .def_readwrite("mask", &MeasurementBase::mask, "Valid components of `value` (1) or ignored ones (0); factors may clear entries (e.g. rejected outliers).")
            .def_readwrite("arclength", &MeasurementBase::arclength, "Where along the robot [m].")
            .def_readwrite("time", &MeasurementBase::time, "When [s].")
            .def("print", &MeasurementBase::print, "Logs the arclength, time and masked value.");

        py::classh<MeasurementQueue>(m, "MeasurementQueue", "A thread-safe queue of measurements, oldest first.")
            .def(py::init<>())
            .def("addMeasurement", &MeasurementQueue::addMeasurement<MeasurementBase>, py::arg("measurement"),
                 "Queues `measurement` (in time order) and calls every registered callback with it.")
            .def("addMeasurementCallback", &MeasurementQueue::addMeasurementCallback, py::arg("callback"),
                 "Calls `callback` with each measurement still queued (oldest first), then with every measurement added from now on. "
                 "It runs with the queue's lock held, so it must not call back into this queue.")
            .def("drainUpTo", &MeasurementQueue::drainUpTo, py::arg("time"), "Removes and returns, oldest first, the queued measurements up to and including `time`.")
            .def("takeAll", &MeasurementQueue::takeAll, "Removes and returns every queued measurement, oldest first, without calling the callbacks.")
            .def("peekAll", &MeasurementQueue::peekAll, "The queued measurements, oldest first, leaving the queue untouched.")
            .def("size", &MeasurementQueue::size, "The number of queued measurements.")
            .def("empty", &MeasurementQueue::empty, "Whether no measurement is queued.")
            .def("lastTime", &MeasurementQueue::lastTime, "The time of the newest queued measurement, or None.")
            .def("frontTime", &MeasurementQueue::frontTime, "The time of the oldest queued measurement (the next drainUpTo returns), or None.");

        py::classh<Model, PyModel<Model>>(m, "Model", "A source of variables and factors for the estimator; override any of its methods.")
            .def(py::init<>())
            .def("extendState", &Model::extendState, py::arg("km"), py::arg("time"), py::arg("current_values"), py::arg("new_values"),
                 "Phase 1: adds the model's new variables up to `time` (initial guesses into `new_values`, keys made with `km`); "
                 "returns the factors that come with them. Must not depend on another model's new variables. Default: none.")
            .def("getNewFactors", &Model::getNewFactors, py::arg("time"), py::arg("new_values"),
                 "Phase 2: returns the model's new factors up to `time`, on variables that exist now. Default: none.")
            .def("onMarginalization", &Model::onMarginalization, py::arg("cutoff"), py::arg("values"),
                 "Called just before every state older than `cutoff` is marginalized, with `values` still holding them "
                 "(in batch mode once, after the solve, with an infinite cutoff). Default: does nothing.")
            .def("lockedValues", &Model::lockedValues, py::return_value_policy::reference_internal,
                 "The model's variables held fixed at these values (gtsam.Values) instead of estimated. Default: none.")
            .def("covarianceKeyGroups", &Model::covarianceKeyGroups, py::arg("values"),
                 "The key groups (lists of keys in `values`) whose joint posterior covariance to extract after each solve. Default: none.");

        py::classh<NodeValuesBase, PyNodeValues>(m, "NodeValuesBase", "The values of a node's variables, one matrix per component.")
            .def(py::init<>())
            .def_readwrite("values", &NodeValuesBase::values, "A copy: assign the whole list to change it.")
            .def("setRandom", &NodeValuesBase::setRandom, "Fills every component with random entries.");

        py::classh<NodeBase, PyNodeBase>(m, "NodeBase", "A node of the state at (arclength, time), holding named variables.")
            .def(py::init<KeyManager &, double, double>(), py::arg("km"), py::arg("arclength"), py::arg("time"),
                 "A subclass's constructor registers its variables with `km` (registerVariable).")
            .def_readonly("arclength", &NodeBase::arclength, "[m]")
            .def_readonly("time", &NodeBase::time, "[s]")
            .def("keys", &NodeBase::keys, "The keys of the node's variables, ordered by variable name.")
            .def("names", &NodeBase::names, "The names of the node's variables, in alphabetical order.")
            .def("key", &NodeBase::key, py::arg("name"), "The key of the variable `name`; raises RuntimeError if there is none.")
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
                 "Inserts `value` for the variable `name` into `values`, as the type it was registered with; raises RuntimeError on a type mismatch.")
            .def("interpolatedKeys", &NodeBase::interpolatedKeys,
                 "[(interpolated key, this node's key)] in the order of the node's tangent vector, letting a prior map a sensor's "
                 "inner factor onto this node (the nearest-node default of StatefulModel). Default: empty, i.e. unsupported.")
            .def("getValuesBase", &NodeBase::getValuesBase, py::arg("values"),
                 "This node's NodeValuesBase read from `values`, or None if the node type does not support it (the default).")
            .def("print", py::overload_cast<const std::string &>(&NodeBase::print, py::const_), py::arg("s") = "",
                 "Prints `s`, the node's location and its variables to stdout.");

        py::classh<State, PyState>(m, "State", "A robot's estimated state: its variables and nodes, extended by one stateful model.")
            .def(py::init<>())
            .def("addStateAtTime", &State::addStateAtTime, py::arg("km"), py::arg("time"), "Adds the state's nodes at `time` (e.g. a node row). Default: does nothing.")
            .def("getNearestNode", &State::getNearestNode, py::arg("arclength"), py::arg("time"),
                 "The node nearest to (arclength, time), or None (the default: no such node, or outside the state's extent).")
            .def("times", &State::times, "The times of the state's node rows, ascending.")
            .def("lockedValues", &State::lockedValues, py::return_value_policy::reference_internal,
                 "The state's locked variables (gtsam.Values): known constants, never estimated.");

        py::classh<StatefulModel, PyStatefulModel<StatefulModel, true>, Model>(
            m, "StatefulModel",
            "A model that owns a state (e.g. a prior): extends it in time (override extendState) and "
            "resolves queries and measurements onto its nodes (by default, the nearest node).")
            .def(py::init<std::shared_ptr<State>>(), py::arg("state"), "Extends `state`; raises ValueError for None.")
            .def("state", &StatefulModel::state, "The state this model extends.")
            .def("getOuterFactor", &StatefulModel::getOuterFactor, py::arg("time"), py::arg("arclength"), py::arg("inner_factor"),
                 "Wraps `inner_factor` (on the node type's interpolated keys) to evaluate it on the state at (time, arclength). "
                 "Default: on the nearest node; None outside the state's extent or without interpolated keys.")
            .def("getMultiPointOuterFactor", &StatefulModel::getMultiPointOuterFactor, py::arg("time"), py::arg("arclengths"), py::arg("inner_factor"),
                 "Wraps `inner_factor`, on point-indexed interpolated keys (point i at (time, arclengths[i]), see InterpolatedKeyAt), "
                 "to evaluate it on the state. Default: a single point through getOuterFactor; None for several (grid priors "
                 "interpolate them jointly). A factor on no point is returned unchanged.")
            .def(
                "query", [](const StatefulModel &self, double time, double arclength, const gtsam::Values &values)
                { return self.query(time, arclength, values); },
                py::arg("time"), py::arg("arclength"), py::arg("values"),
                "The node values at (time, arclength), or None if unresolvable (outside the extent, or a needed node not in `values`). "
                "A Python override may return (node values, conditional covariance).")
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
            .def(py::init<pStateful>(), py::arg("stateful"), "Measures the state of `stateful` (e.g. a prior); raises RuntimeError for None.")
            .def("getFactor", &SensorBase::getFactor, py::arg("measurement"),
                 "The inner factor of `measurement`, on (some of) the node type's interpolated keys, with its own noise model; None drops it.")
            .def("stateful", &SensorBase::stateful, "The stateful model whose state this sensor measures.");

        py::classh<ActuatorBase, PyActuator, StatefulModel, MeasurementQueue>(
            m, "ActuatorBase", "An input acting along the robot: each measured input becomes a variable shared by its factors (override getFactors).")
            .def(py::init([](pStateful stateful, std::vector<double> arclengths, unsigned int input_dim, char input_symbol)
                          { return new PyActuator(std::move(stateful), std::move(arclengths), input_dim, static_cast<unsigned char>(input_symbol)); }),
                 py::arg("stateful"), py::arg("arclengths"), py::arg("input_dim"), py::arg("input_symbol") = 'u',
                 "Acts on the state of `stateful` at `arclengths` [m] with inputs of size `input_dim`, keyed with `input_symbol`. "
                 "Raises RuntimeError for no stateful model, ValueError for no arclengths or a zero input_dim.")
            .def("getFactors", &ActuatorBase::getFactors, py::arg("measurement"), py::arg("input_key"),
                 "The inner factors of `measurement`'s input (the variable `input_key`), the one method to override: each on the "
                 "node type's interpolated keys at any of the actuator's points (e.g. wnoa.InterpolatedKeys.Pose(i) at arclengths()[i]), "
                 "`input_key` and any other variable, with its own noise model. A factor per point acts at each independently, one "
                 "factor on several points acts on them jointly, one on no point constrains the input alone. Return [] or None for none.")
            .def("stateful", &ActuatorBase::stateful, "The stateful model whose state the input acts on.")
            .def("arclengths", &ActuatorBase::arclengths, "Where the input acts [m].")
            .def("inputDim", &ActuatorBase::inputDim, "The size of every input.")
            .def("inputKeys", &ActuatorBase::inputKeys, "[(measurement time, key)] of every input added so far, oldest first, including marginalized ones.");

        py::classh<Robot>(m, "Robot", "A convenience list of the models making up one robot (e.g. its prior and sensors), for Estimator.addRobot.")
            .def(py::init<>())
            .def("addModel", &Robot::addModel, py::arg("model"), "Appends `model`; raises ValueError for None.")
            .def("models", &Robot::models, "The models added so far, in order.");
    }
} // namespace SpacetimePython
