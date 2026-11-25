#ifndef ENDEFFECTORSTATE_HPP
#define ENDEFFECTORSTATE_HPP

#include <memory>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/types/ActuatorBase.hpp>
#include <spacetime/estimator/types/ContinuumRobot.hpp>
#include <spacetime/estimator/types/InterpolatedKeys.hpp>
#include <spacetime/estimator/types/Robot.hpp>
#include <spacetime/estimator/types/SensorBase.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

namespace Spacetime::Joints
{
    template <typename Pose>
    struct EndEffectorKeys;

    /**
     * @brief Values of an end-effector node: pose T_ib and generalized (body) velocity
     *        varpi, in GTSAM [rotational, translational] order. Stored in that order in
     *        Types::NodeValuesBase::values (pose as its matrix).
     */
    template <typename Pose>
    struct EndEffectorNodeValues : public Types::NodeValuesBase
    {
        using Base = Types::NodeValuesBase;                                    ///< Type-erased node values.
        using TangentVector = typename gtsam::traits<Pose>::TangentVector; ///< Tangent vector of Pose ([rot, trans]).
        using PoseType = Pose;                                             ///< The pose type.
        using InterpolatedKeys = EndEffectorKeys<Pose>;                    ///< Placeholder keys of an interpolated node.
        static constexpr unsigned int dim = 2 * gtsam::traits<Pose>::dimension; ///< Tangent dimension of the node (pose + velocity).

        /// Identity pose, at rest.
        EndEffectorNodeValues()
        {
            values.resize(2);
            pose(Pose());
            varpi(TangentVector::Zero());
        }

        /// Values holding `pose_value` and `varpi_value`.
        EndEffectorNodeValues(const Pose &pose_value, const TangentVector &varpi_value)
        {
            values.resize(2);
            pose(pose_value);
            varpi(varpi_value);
        }

        EndEffectorNodeValues(const EndEffectorNodeValues &other) = default;

        /// The pose T_ib.
        Pose pose() const { return Pose(values.at(0)); }
        /// Sets the pose T_ib.
        void pose(const Pose &value) { values[0] = value.matrix(); }
        /// The body twist varpi ([rot, trans]).
        TangentVector varpi() const { return values.at(1); }
        /// Sets the body twist varpi ([rot, trans]).
        void varpi(const TangentVector &value) { values[1] = value; }

        /// Random pose (Exp of a random tangent) and velocity.
        void setRandom() override
        {
            pose(gtsam::traits<Pose>::Expmap(TangentVector::Random()));
            varpi(TangentVector::Random());
        }

        /// Copy from another EndEffectorNodeValues; throws std::runtime_error for any other NodeValuesBase type.
        EndEffectorNodeValues &operator=(const Base &other) override
        {
            if (!dynamic_cast<const EndEffectorNodeValues *>(&other))
                throw std::runtime_error("EndEffectorNodeValues::operator= received incompatible node values type.");
            Base::operator=(other);
            return *this;
        }
        /// Copy assignment.
        EndEffectorNodeValues &operator=(const EndEffectorNodeValues &other) { return static_cast<EndEffectorNodeValues &>(Base::operator=(other)); }
    };

    /// An end-effector node: pose ('p') and velocity ('v') variables, tangent order [pose, velocity].
    template <typename Pose>
    struct EndEffectorNode : public Types::NodeBase
    {
        static constexpr unsigned int dim = 2 * gtsam::traits<Pose>::dimension; ///< Tangent dimension (pose + velocity).
        using ValuesType = EndEffectorNodeValues<Pose>;                        ///< The node's values.
        using TangentVector = typename gtsam::traits<Pose>::TangentVector; ///< Tangent vector of Pose ([rot, trans]).

        /// Register this node's pose ('p') and velocity ('v') variables with `km`.
        EndEffectorNode(KeyManager &km, double arclength, double time) : Types::NodeBase(km, arclength, time)
        {
            poseKey_ = this->registerVariable<Pose>(km, "pose", 'p');
            velocityKey_ = this->registerVariable<TangentVector>(km, "velocity", 'v');
        }
        /// Key of the pose variable ('p').
        gtsam::Key poseKey() const { return poseKey_; }
        /// Key of the velocity variable ('v').
        gtsam::Key velocityKey() const { return velocityKey_; }

        /// (interpolated key, this node's key) pairs for pose and velocity (see EndEffectorKeys).
        std::vector<std::pair<gtsam::Key, gtsam::Key>> interpolatedKeys() const override
        {
            return {{EndEffectorKeys<Pose>::Pose(), poseKey_}, {EndEffectorKeys<Pose>::Velocity(), velocityKey_}};
        }

        /// getValues(), type-erased.
        std::shared_ptr<Types::NodeValuesBase> getValuesBase(const gtsam::Values &values) const override
        {
            return std::make_shared<ValuesType>(getValues(values));
        }

        /// This node's values read from `values` (throws if a variable is absent).
        ValuesType getValues(const gtsam::Values &values) const
        {
            return ValuesType(values.at<Pose>(poseKey_), values.at<TangentVector>(velocityKey_));
        }
        /// Insert `node_values` into `values` at this node's keys.
        void setValues(gtsam::Values &values, const ValuesType &node_values) const
        {
            this->insert("pose", node_values.pose(), values);
            this->insert("velocity", node_values.varpi(), values);
        }

    private:
        gtsam::Key poseKey_, velocityKey_;
    };

    /**
     * @brief The end effector's interpolated keys (see Types::InterpolatedKeyMap): pose and
     *        velocity, as rows [0, dim) and [dim, 2 dim) of the node's tangent vector.
     */
    template <typename PoseT>
    struct EndEffectorKeys
    {
        using ValuesType = EndEffectorNodeValues<PoseT>;                ///< The values the keys index.
        static constexpr int dim = gtsam::traits<PoseT>::dimension; ///< Tangent dimension of the pose.

        /// Placeholder key of the interpolated pose.
        static gtsam::Key Pose() { return gtsam::Symbol('q', 0); }
        /// Placeholder key of the interpolated velocity.
        static gtsam::Key Velocity() { return gtsam::Symbol('q', 1); }
        /// Every placeholder key, in tangent order.
        static gtsam::KeyVector keys() { return {Pose(), Velocity()}; }
        /// The interpolated keys at the `point`-th arclength of a multi-point factor (see Types::InterpolatedKeyAt).
        static gtsam::Key Pose(std::size_t point) { return Types::InterpolatedKeyAt(Pose(), point); }
        static gtsam::Key Velocity(std::size_t point) { return Types::InterpolatedKeyAt(Velocity(), point); }
        static gtsam::KeyVector keys(std::size_t point) { return Types::InterpolatedKeysAt(keys(), point); }

        /// Inserts the piece of `node` that `key` names into `values`. @throws std::runtime_error for any other key.
        static void insert(gtsam::Key key, const ValuesType &node, gtsam::Values &values)
        {
            if (key == Pose())
                values.insert(key, node.pose());
            else if (key == Velocity())
                values.insert(key, node.varpi());
            else
                throw std::runtime_error("EndEffectorKeys::insert: key is not one of the end effector's interpolated keys.");
        }

        /// (offset, dim) of `key`'s rows in the node's tangent vector. @throws std::runtime_error for any other key.
        static std::pair<int, int> rowRange(gtsam::Key key)
        {
            if (key == Pose())
                return {0, dim};
            if (key == Velocity())
                return {dim, dim};
            throw std::runtime_error("EndEffectorKeys::rowRange: key is not one of the end effector's interpolated keys.");
        }
    };
} // namespace Spacetime::Joints

#endif // ENDEFFECTORSTATE_HPP
