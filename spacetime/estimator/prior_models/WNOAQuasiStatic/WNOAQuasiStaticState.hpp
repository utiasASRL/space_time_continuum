#ifndef WNOAQUASISTATICSTATE_HPP
#define WNOAQUASISTATICSTATE_HPP

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/prior_models/StraightStrain.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/types/ActuatorBase.hpp>
#include <spacetime/estimator/types/ContinuumRobot.hpp>
#include <spacetime/estimator/types/InterpolatedKeys.hpp>
#include <spacetime/estimator/types/Robot.hpp>
#include <spacetime/estimator/types/SensorBase.hpp>

namespace Spacetime::Priors::WNOAQuasiStatic
{
    // Forward declaration: the concrete key layout lives in WNOAQuasiStatic/InterpolatedKeys.hpp;
    // NodeValues<Pose> only needs to name it here, not use it, so this avoids a cycle.
    template <typename Pose>
    struct InterpolatedKeys;

    /**
     * @brief Values of a quasi-static node: pose T_ib and generalized strain epsilon (GTSAM
     *        [rotational, translational] order). The quasi-static prior has no time
     *        dimension, so no velocity is part of the state.
     * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
     */
    template <typename Pose>
    struct NodeValues : public Types::NodeValuesBase
    {
        using Base = Types::NodeValuesBase;
        using TangentVector = typename gtsam::traits<Pose>::TangentVector;
        using PoseType = Pose;
        using InterpolatedKeys = WNOAQuasiStatic::InterpolatedKeys<Pose>; ///< This node type's interpolated-key layout.
        /// Tangent dimension: pose + strain.
        static constexpr unsigned int dim = 2 * gtsam::traits<Pose>::dimension;

        /// Identity pose and straight strain (see Priors::straightStrain: [0, 0, 0, 1, 0, 0]
        /// for gtsam::Pose3, the backbone along body x).
        NodeValues() : NodeValues(Pose(), straightStrain<Pose>()) {}

        /// Values holding pose `pose_value` (T_ib) and strain `epsilon_value`.
        NodeValues(const Pose &pose_value, const TangentVector &epsilon_value)
        {
            values.resize(2);
            pose(pose_value);
            epsilon(epsilon_value);
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

        /// Generalized strain, [rotational, translational].
        TangentVector epsilon() const
        {
            return values.at(1);
        }

        /// Set the generalized strain.
        void epsilon(const TangentVector &value)
        {
            values[1] = value;
        }

        /// Random pose (Exp of a random tangent) and strain.
        void setRandom() override
        {
            pose(gtsam::traits<Pose>::Expmap(TangentVector::Random()));
            epsilon(TangentVector::Random());
        }

        /// Copy from another WNOAQuasiStatic::NodeValues; throws std::runtime_error for any other NodeValuesBase type.
        NodeValues &operator=(const Base &other) override
        {
            const auto *typed_other = dynamic_cast<const NodeValues *>(&other);
            if (!typed_other)
                throw std::runtime_error("WNOAQuasiStatic::NodeValues::operator= received incompatible node values type.");

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
     * @brief A quasi-static node at (arclength, time): pose and strain variables, with tangent order [pose, strain].
     * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
     */
    template <typename Pose>
    struct Node : public Types::NodeBase
    {
        /// Tangent dimension: pose + strain.
        static constexpr unsigned int dim = 2 * gtsam::traits<Pose>::dimension;
        using ValuesType = NodeValues<Pose>; ///< This node's values.
        using TangentVector = typename gtsam::traits<Pose>::TangentVector;

        /// Register this node's pose ('p') and strain ('s') variables with `km`.
        Node(KeyManager &km, double arclength, double time) : Types::NodeBase(km, arclength, time)
        {
            poseKey_ = this->registerVariable<Pose>(km, "pose", 'p');
            strainKey_ = this->registerVariable<TangentVector>(km, "strain", 's');
        }
        /// Key of the pose variable.
        gtsam::Key poseKey() const { return poseKey_; }
        /// Key of the strain variable.
        gtsam::Key strainKey() const { return strainKey_; }

        /// (interpolated key, this node's key) pairs for pose and strain (see WNOAQuasiStatic::InterpolatedKeys).
        std::vector<std::pair<gtsam::Key, gtsam::Key>> interpolatedKeys() const override
        {
            // InterpolatedKeys is included at the end of this file (see there).
            using Keys = InterpolatedKeys<Pose>;
            return {{Keys::Pose(), poseKey_}, {Keys::Strain(), strainKey_}};
        }

        /// This node's values read from `values` (throws if a variable is absent).
        std::shared_ptr<Types::NodeValuesBase> getValuesBase(const gtsam::Values &values) const override
        {
            return std::make_shared<ValuesType>(getValues(values));
        }

        /// Typed counterpart of getValuesBase.
        ValuesType getValues(const gtsam::Values &values) const
        {
            return ValuesType(values.at<Pose>(poseKey_), values.at<TangentVector>(strainKey_));
        }
        /// Insert `node_values` into `values` at this node's keys.
        void setValues(gtsam::Values &values, const ValuesType &node_values) const
        {
            this->insert("pose", node_values.pose(), values);
            this->insert("strain", node_values.epsilon(), values);
        }

    private:
        gtsam::Key poseKey_, strainKey_;
    };
} // namespace Spacetime::Priors::WNOAQuasiStatic

// Node::interpolatedKeys needs InterpolatedKeys complete. It includes this file, so it can
// only come after Node is defined; its include guard breaks the cycle.
#include <spacetime/estimator/prior_models/WNOAQuasiStatic/InterpolatedKeys.hpp>

#endif // WNOAQUASISTATICSTATE_HPP
