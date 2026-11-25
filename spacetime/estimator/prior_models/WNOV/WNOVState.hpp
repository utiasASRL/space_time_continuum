#ifndef WNOVSTATE_HPP
#define WNOVSTATE_HPP

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/types/ActuatorBase.hpp>
#include <spacetime/estimator/types/ContinuumRobot.hpp>
#include <spacetime/estimator/types/InterpolatedKeys.hpp>
#include <spacetime/estimator/types/Robot.hpp>
#include <spacetime/estimator/types/SensorBase.hpp>

namespace Spacetime::Priors::WNOV
{
    // Forward declaration: the concrete key layout lives in WNOV/InterpolatedKeys.hpp;
    // NodeValues<Pose> only needs to name it here, not use it, so this avoids a cycle.
    template <typename Pose>
    struct InterpolatedKeys;

    /**
     * @brief Values of a WNOV node: its pose alone. Under the white-noise-on-velocity
     *        prior the pose is a random walk in arclength and time, so no strain or
     *        velocity is part of the state.
     * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
     */
    template <typename Pose>
    struct NodeValues : public Types::NodeValuesBase
    {
        using Base = Types::NodeValuesBase;
        using TangentVector = typename gtsam::traits<Pose>::TangentVector;
        using PoseType = Pose;
        using InterpolatedKeys = WNOV::InterpolatedKeys<Pose>; ///< This node type's interpolated-key layout.
        /// Tangent dimension (the pose's).
        static constexpr unsigned int dim = gtsam::traits<Pose>::dimension;

        /// Identity pose.
        NodeValues()
        {
            values.resize(1);
            pose(Pose());
        }

        /// Values holding pose `value` (T_ib).
        explicit NodeValues(const Pose &value)
        {
            values.resize(1);
            pose(value);
        }

        /// Copy constructor.
        NodeValues(const NodeValues &other) = default;

        /// Pose T_ib.
        Pose pose() const
        {
            return Pose(values.at(0));
        }

        /// Set the pose T_ib.
        void pose(const Pose &value)
        {
            values[0] = value.matrix();
        }

        /// Random pose (Exp of a random tangent).
        void setRandom() override
        {
            pose(gtsam::traits<Pose>::Expmap(TangentVector::Random()));
        }

        /// Copy from another WNOV::NodeValues; throws std::runtime_error for any other NodeValuesBase type.
        NodeValues &operator=(const Base &other) override
        {
            const auto *typed_other = dynamic_cast<const NodeValues *>(&other);
            if (!typed_other)
                throw std::runtime_error("WNOV::NodeValues::operator= received incompatible node values type.");

            Base::operator=(other);
            return *this;
        }
        /// Copy assignment.
        NodeValues &operator=(const NodeValues &other)
        {
            return static_cast<NodeValues &>(Base::operator=(other));
        }
    };

    /**
     * @brief A WNOV node: a single pose variable at (arclength, time).
     * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
     */
    template <typename Pose>
    struct Node : public Types::NodeBase
    {
        /// Tangent dimension (the pose's).
        static constexpr unsigned int dim = gtsam::traits<Pose>::dimension;
        using ValuesType = NodeValues<Pose>; ///< This node's values.

        /// Register this node's pose variable ('p') with `km`.
        Node(KeyManager &km, double arclength, double time) : Types::NodeBase(km, arclength, time)
        {
            poseKey_ = this->registerVariable<Pose>(km, "pose", 'p');
        }
        /// Key of the pose variable.
        gtsam::Key poseKey() const { return poseKey_; }

        /// The single (interpolated pose key, this node's pose key) pair (see WNOV::InterpolatedKeys).
        std::vector<std::pair<gtsam::Key, gtsam::Key>> interpolatedKeys() const override
        {
            // InterpolatedKeys is included at the end of this file (see there).
            return {{InterpolatedKeys<Pose>::Pose(), poseKey_}};
        }

        /// This node's values read from `values` (throws if the pose is absent).
        std::shared_ptr<Types::NodeValuesBase> getValuesBase(const gtsam::Values &values) const override
        {
            return std::make_shared<ValuesType>(values.at<Pose>(poseKey_));
        }

        /// Typed counterpart of getValuesBase.
        ValuesType getValues(const gtsam::Values &values) const
        {
            return ValuesType(values.at<Pose>(poseKey_));
        }
        /// Insert `node_values` into `values` at this node's pose key.
        void setValues(gtsam::Values &values, const ValuesType &node_values) const
        {
            this->insert("pose", node_values.pose(), values);
        }

    private:
        gtsam::Key poseKey_;
    };
} // namespace Spacetime::Priors::WNOV

// Node::interpolatedKeys needs InterpolatedKeys complete. It includes this file, so it can
// only come after Node is defined; its include guard breaks the cycle.
#include <spacetime/estimator/prior_models/WNOV/InterpolatedKeys.hpp>

#endif // WNOVSTATE_HPP
