#ifndef SPACETIME_PYTHON_COMMON_HPP
#define SPACETIME_PYTHON_COMMON_HPP

/**
 * @file common.hpp
 * @brief Shared by every binding source: the Pose3 instantiations the bindings expose, and
 *        the trampolines that let Python subclasses override the library's virtual methods.
 *
 * Holders: the library's own classes are bound with py::smart_holder, so a Python subclass
 * held only by C++ (e.g. a sensor given to Estimator::addSensor) stays alive, and
 * enable_shared_from_this keeps working. GTSAM's classes keep the
 * std::shared_ptr holder of GTSAM's own Python module, which registers them.
 *
 * Threads: the standalone session calls models from its estimator thread, with the GIL
 * released by Session.join(); every trampoline takes the GIL before looking for an override.
 */

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h> // before the library: gtsam/nonlinear/WnoaFactor.h uses it without including it
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/joint_models/EndEffector.hpp>
#include <spacetime/estimator/joint_models/Joint.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic.hpp>
#include <spacetime/estimator/types/ActuatorBase.hpp>
#include <spacetime/estimator/types/ContinuumRobot.hpp>
#include <spacetime/estimator/types/Model.hpp>
#include <spacetime/estimator/types/Robot.hpp>
#include <spacetime/estimator/types/SensorBase.hpp>
#include <spacetime/estimator/types/StatefulModel.hpp>
#include <spacetime/estimator/types/States.hpp>

// After the library: Python's headers define macros that clash with GTSAM's.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace SpacetimePython
{
    using Pose = gtsam::Pose3;
    using Vector6 = Eigen::Matrix<double, 6, 1>;
    using Factors = Spacetime::Types::Model::Factors;
    using StatefulModel = Spacetime::Types::StatefulModel<>;
    using pStateful = std::shared_ptr<StatefulModel>;

    using WnoaPrior = Spacetime::Priors::WNOA::Prior<Pose>;
    using WnoaNode = WnoaPrior::NodeType;
    using WnoaValues = WnoaPrior::ValuesType;
    using WnoaState = WnoaPrior::StateType;

    using WnovPrior = Spacetime::Priors::WNOV::Prior<Pose>;
    using WnovNode = WnovPrior::NodeType;
    using WnovValues = WnovPrior::ValuesType;
    using WnovState = WnovPrior::StateType;

    using QuasiStaticPrior = Spacetime::Priors::WNOAQuasiStatic::Prior<Pose>;
    using QuasiStaticNode = QuasiStaticPrior::NodeType;
    using QuasiStaticValues = QuasiStaticPrior::ValuesType;
    using QuasiStaticState = QuasiStaticPrior::StateType;

    using EndEffector = Spacetime::Joints::EndEffector<Pose>;
    using EndEffectorPrior = EndEffector::PriorType;
    using EndEffectorNode = EndEffector::NodeType;
    using EndEffectorValues = Spacetime::Joints::EndEffectorValues<Pose>;
    using EndEffectorState = Spacetime::Types::ContinuumRobotState<EndEffectorNode>;

    using RobotJoint = Spacetime::Joints::Joint<WnoaPrior, WnoaPrior>;
    using EndEffectorJoint = Spacetime::Joints::Joint<WnoaPrior, EndEffectorPrior>;

    /// A Python reference to `value`, not a copy: overrides see (and may modify) the caller's object.
    template <typename T>
    py::object ref(const T &value)
    {
        return py::cast(const_cast<T *>(&value), py::return_value_policy::reference);
    }

    inline std::runtime_error pureVirtual(const std::string &name)
    {
        return std::runtime_error("Tried to call pure virtual function \"" + name + "\": override it in the Python subclass.");
    }

/// Returns the Python override `name` of `this` (looked up as a `Base`) called with the arguments, if there is one.
#define SPACETIME_OVERRIDE(Return, Base, name, ...)                                                   \
    do                                                                                                \
    {                                                                                                 \
        py::gil_scoped_acquire gil;                                                                   \
        if (py::function py_override = py::get_override(static_cast<const Base *>(this), name))        \
            return py_override(__VA_ARGS__).template cast<Return>();                                     \
    } while (false)

    /// Python overrides of the Types::Model virtuals on top of `Base` (Model or one of its children).
    template <typename Base, bool PureExtendState = false>
    struct PyModel : Base, py::trampoline_self_life_support
    {
        using Base::Base;

        Factors extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override
        {
            SPACETIME_OVERRIDE(Factors, Base, "extendState", ref(km), time, ref(current_values), ref(new_values));
            if constexpr (PureExtendState)
                throw pureVirtual("extendState");
            else
                return Base::extendState(km, time, current_values, new_values);
        }

        Factors getNewFactors(double time, gtsam::Values &new_values) override
        {
            SPACETIME_OVERRIDE(Factors, Base, "getNewFactors", time, ref(new_values));
            return Base::getNewFactors(time, new_values);
        }

        void onMarginalization(double cutoff, const gtsam::Values &values) override
        {
            SPACETIME_OVERRIDE(void, Base, "onMarginalization", cutoff, ref(values));
            Base::onMarginalization(cutoff, values);
        }

        /// A Python override's values are kept here, since the caller receives a reference.
        const gtsam::Values &lockedValues() const override
        {
            {
                py::gil_scoped_acquire gil;
                if (py::function py_override = py::get_override(static_cast<const Base *>(this), "lockedValues"))
                {
                    m_locked_values = py_override().template cast<gtsam::Values>();
                    return m_locked_values;
                }
            }
            return Base::lockedValues();
        }

        std::vector<gtsam::KeyVector> covarianceKeyGroups(const gtsam::Values &values) const override
        {
            SPACETIME_OVERRIDE(std::vector<gtsam::KeyVector>, Base, "covarianceKeyGroups", ref(values));
            return Base::covarianceKeyGroups(values);
        }

    private:
        mutable gtsam::Values m_locked_values;
    };

    /// Python overrides of the Types::StatefulModel virtuals (and the Model ones) on top of `Base`.
    template <typename Base, bool PureExtendState = false>
    struct PyStatefulModel : PyModel<Base, PureExtendState>
    {
        using PyModel<Base, PureExtendState>::PyModel;

        std::shared_ptr<gtsam::NonlinearFactor> getOuterFactor(double time, double arclength,
                                                               const gtsam::NoiseModelFactor::shared_ptr &inner_factor) const override
        {
            SPACETIME_OVERRIDE(std::shared_ptr<gtsam::NonlinearFactor>, Base, "getOuterFactor", time, arclength, inner_factor);
            return Base::getOuterFactor(time, arclength, inner_factor);
        }

        /**
         * A Python `query(time, arclength, values)` returns the node values, or a tuple of them and
         * their conditional covariance. It gives no posterior covariance: that output is left empty.
         */
        std::shared_ptr<Spacetime::Types::NodeValues> query(double time, double arclength, const gtsam::Values &values,
                                                            Eigen::MatrixXd *conditionalCovariance = nullptr,
                                                            const Spacetime::PosteriorCovariance *posterior = nullptr,
                                                            Eigen::MatrixXd *posteriorCovariance = nullptr) const override
        {
            {
                py::gil_scoped_acquire gil;
                if (py::function py_override = py::get_override(static_cast<const Base *>(this), "query"))
                {
                    const py::object result = py_override(time, arclength, ref(values));
                    std::shared_ptr<Spacetime::Types::NodeValues> node;
                    if (py::isinstance<py::tuple>(result))
                    {
                        const auto items = result.cast<py::tuple>();
                        node = items[0].cast<std::shared_ptr<Spacetime::Types::NodeValues>>();
                        if (conditionalCovariance)
                            *conditionalCovariance = items[1].cast<Eigen::MatrixXd>();
                    }
                    else
                    {
                        node = result.cast<std::shared_ptr<Spacetime::Types::NodeValues>>();
                        if (conditionalCovariance)
                            conditionalCovariance->resize(0, 0);
                    }
                    if (posteriorCovariance)
                        posteriorCovariance->resize(0, 0);
                    return node;
                }
            }
            return Base::query(time, arclength, values, conditionalCovariance, posterior, posteriorCovariance);
        }
    };

    /// Python overrides of SensorBase::getFactor (and the Model virtuals) on top of `Base`.
    template <typename Base, bool PureGetFactor = false>
    struct PySensor : PyModel<Base>
    {
        using PyModel<Base>::PyModel;

        gtsam::NoiseModelFactor::shared_ptr getFactor(const Spacetime::Types::SensorBase::pMeasurementType measurement) const override
        {
            SPACETIME_OVERRIDE(gtsam::NoiseModelFactor::shared_ptr, Base, "getFactor", measurement);
            if constexpr (PureGetFactor)
                throw pureVirtual("SensorBase::getFactor");
            else
                return Base::getFactor(measurement);
        }
    };

    /// Python overrides of ActuatorBase::getFactor (and the Model virtuals).
    struct PyActuator : PyModel<Spacetime::Types::ActuatorBase>
    {
        using PyModel<Spacetime::Types::ActuatorBase>::PyModel;

        gtsam::NoiseModelFactor::shared_ptr getFactor(const Spacetime::Types::MeasurementBase &measurement, gtsam::Key input_key,
                                                      double arclength) const override
        {
            SPACETIME_OVERRIDE(gtsam::NoiseModelFactor::shared_ptr, Spacetime::Types::ActuatorBase, "getFactor", ref(measurement), input_key,
                               arclength);
            throw pureVirtual("ActuatorBase::getFactor");
        }
    };

    struct PyNodeValues : Spacetime::Types::NodeValues, py::trampoline_self_life_support
    {
        using Spacetime::Types::NodeValues::NodeValues;

        void setRandom() override
        {
            SPACETIME_OVERRIDE(void, Spacetime::Types::NodeValues, "setRandom");
            Spacetime::Types::NodeValues::setRandom();
        }
    };

    struct PyNodeBase : Spacetime::Types::NodeBase, py::trampoline_self_life_support
    {
        using Spacetime::Types::NodeBase::NodeBase;
        using Spacetime::Types::NodeBase::registerVariable; // protected: for Python subclasses' constructors
        using KeyPairs = std::vector<std::pair<gtsam::Key, gtsam::Key>>;

        KeyPairs interpolatedKeys() const override
        {
            SPACETIME_OVERRIDE(KeyPairs, Spacetime::Types::NodeBase, "interpolatedKeys");
            return Spacetime::Types::NodeBase::interpolatedKeys();
        }

        std::shared_ptr<Spacetime::Types::NodeValues> getValuesBase(const gtsam::Values &values) const override
        {
            SPACETIME_OVERRIDE(std::shared_ptr<Spacetime::Types::NodeValues>, Spacetime::Types::NodeBase, "getValuesBase", ref(values));
            return Spacetime::Types::NodeBase::getValuesBase(values);
        }
    };

    struct PyState : Spacetime::Types::State, py::trampoline_self_life_support
    {
        using Spacetime::Types::State::State;

        void addStateAtTime(KeyManager &km, double time) override
        {
            SPACETIME_OVERRIDE(void, Spacetime::Types::State, "addStateAtTime", ref(km), time);
            Spacetime::Types::State::addStateAtTime(km, time);
        }

        std::shared_ptr<Spacetime::Types::NodeBase> getNearestNode(double arclength, double time) const override
        {
            SPACETIME_OVERRIDE(std::shared_ptr<Spacetime::Types::NodeBase>, Spacetime::Types::State, "getNearestNode", arclength, time);
            return Spacetime::Types::State::getNearestNode(arclength, time);
        }
    };

    /// The python type `kind` stands for, to register a variable of it: gtsam.Pose3, gtsam.Rot3 or numpy.ndarray (a gtsam Vector).
    template <typename Register>
    gtsam::Key registerByKind(const py::object &kind, Register &&register_variable)
    {
        if (kind.is(py::type::of<gtsam::Pose3>()))
            return register_variable(static_cast<gtsam::Pose3 *>(nullptr));
        if (kind.is(py::type::of<gtsam::Rot3>()))
            return register_variable(static_cast<gtsam::Rot3 *>(nullptr));
        if (kind.is(py::module_::import("numpy").attr("ndarray")))
            return register_variable(static_cast<gtsam::Vector *>(nullptr));
        throw py::type_error("Unsupported variable type: use gtsam.Pose3, gtsam.Rot3 or numpy.ndarray (a gtsam Vector).");
    }

    // Binding entry points, one per source file.
    void bindTypes(py::module_ &m);
    void bindPriors(py::module_ &m);
    void bindJoints(py::module_ &m);
    void bindSensors(py::module_ &m);
    void bindEstimator(py::module_ &m);
    void bindInterface(py::module_ &m);
    void bindUtilities(py::module_ &m);
#ifdef SPACETIME_PYTHON_VISUALIZER
    void bindVisualizer(py::module_ &m);
#endif
#ifdef SPACETIME_PYTHON_ROS2
    void bindRos2(py::module_ &m);
    void bindRviz(py::module_ &m);
#endif
} // namespace SpacetimePython

#endif // SPACETIME_PYTHON_COMMON_HPP
