#ifndef WNOASTATE_HPP
#define WNOASTATE_HPP

#include <iostream>
#include <cassert>

#include <Eigen/Core>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>

#include <spacetime/estimator/utilities/functions.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/Types.hpp>

namespace Spacetime::Priors::WNOA
{
    // Forward declaration: the concrete key layout (pose/strain/velocity) lives in
    // WNOA/InterpolatedKeys.hpp, alongside its own generic-interface documentation;
    // Values<Pose> only needs to name it here, not use it, so this avoids a cycle.
    template <typename Pose>
    struct InterpolatedKeys;

    /**
     * @brief Values of a WNOA node: pose T_ib, generalized strain epsilon and generalized
     *        velocity varpi, the latter two in GTSAM [rotational, translational] order.
     *        Stored in that order in Types::NodeValues::values (pose as its matrix).
     * @tparam Pose The pose Lie group (e.g. gtsam::Pose3, or gtsam::Pose2 in the plane).
     */
    template <typename Pose>
    struct Values : public Types::NodeValues
    {
        using Base = Types::NodeValues;
        using TangentVector = typename gtsam::traits<Pose>::TangentVector; ///< Strain/velocity type, [rot, trans].
        using PoseType = Pose;
        using InterpolatedKeys = WNOA::InterpolatedKeys<Pose>; ///< This node type's interpolated-key layout.
        /// Tangent dimension: pose + strain + velocity.
        static constexpr unsigned int dim = Pose::dimension + 2 * TangentVector::RowsAtCompileTime;

        /// Identity pose, zero strain and velocity.
        Values()
        {
            values.resize(3);
            pose(Pose());
            epsilon(TangentVector::Zero());
            varpi(TangentVector::Zero());
        }

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

        /// Set the generalized strain, [rotational, translational].
        void epsilon(const TangentVector &value)
        {
            values[1] = value;
        }

        /// Generalized velocity, [rotational, translational].
        TangentVector varpi() const
        {
            return values.at(2);
        }

        /// Set the generalized velocity, [rotational, translational].
        void varpi(const TangentVector &value)
        {
            values[2] = value;
        }

        /// Random pose (Exp of a random tangent), strain and velocity.
        void setRandom() override 
        {
            pose(gtsam::traits<Pose>::Expmap(TangentVector::Random()));
            epsilon(TangentVector::Random());
            varpi(TangentVector::Random());
        }

        /// Copy from another WNOA::Values; throws std::runtime_error for any other NodeValues type.
        Values& operator=(const Base &other) override
        {
            const auto *typed_other = dynamic_cast<const Values *>(&other);
            if (!typed_other)
                throw std::runtime_error("WNOA::Values::operator= received incompatible node values type.");

            Base::operator=(other);
            return *this;
        }
        /// Copy assignment.
        Values& operator=(const Values &other)
        {
            return static_cast<Values&>(Base::operator=(other));
        }
    };

    // Notes: https://borglab.github.io/gtsam/manifold/#id-3-how-the-traits-specialization-works
    /**
     * @brief A WNOA node at (arclength, time): pose, strain and velocity variables, with
     *        tangent (and Jacobian) order [pose, strain, velocity].
     * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
     */
    template <typename Pose>
    struct Node : public Types::NodeBase
    {
        /// Tangent dimension: pose + strain + velocity.
        static constexpr unsigned int dim = gtsam::traits<Pose>::dimension * 3;
        using ValuesType = Values<Pose>; ///< This node's values.
        using TangentVector = typename gtsam::traits<Pose>::TangentVector;

        // Node specific functions
        /// Register this node's pose ('p'), strain ('s') and velocity ('v') variables with `km`.
        Node(KeyManager &km, double arclength, double time) : Types::NodeBase(km, arclength, time)
        {
            poseKey_ = this->registerVariable<Pose>(km, "pose", 'p');
            strainKey_ = this->registerVariable<TangentVector>(km, "strain", 's');
            velocityKey_ = this->registerVariable<TangentVector>(km, "velocity", 'v');
        }
        /// Key of the pose variable.
        gtsam::Key poseKey() const { return poseKey_; }
        /// Key of the strain variable.
        gtsam::Key strainKey() const { return strainKey_; }
        /// Key of the velocity variable.
        gtsam::Key velocityKey() const { return velocityKey_; }

        /// (interpolated key, this node's key) pairs for pose, strain and velocity (see WNOA::InterpolatedKeys).
        std::vector<std::pair<gtsam::Key, gtsam::Key>> interpolatedKeys() const override
        {
            // InterpolatedKeys is included at the end of this file (see there).
            using Keys = InterpolatedKeys<Pose>;
            return {{Keys::Pose(), poseKey_}, {Keys::Strain(), strainKey_}, {Keys::Velocity(), velocityKey_}};
        }

        /// This node's values read from `values` (throws if a variable is absent).
        std::shared_ptr<Types::NodeValues> getValuesBase(const gtsam::Values &values) const override
        {
            std::shared_ptr<ValuesType> ret = std::make_shared<ValuesType>();
            ret->pose(values.at<Pose>(poseKey_));
            ret->epsilon(values.at<TangentVector>(strainKey_));
            ret->varpi(values.at<TangentVector>(velocityKey_));
            return ret;
        }

        // WNOA state specific functions
        /// Typed counterpart of getValuesBase.
        ValuesType getValues(const gtsam::Values &values) const
        {
            return dynamic_cast<ValuesType&>(*getValuesBase(values));
        }
        /// Insert `node_values` into `values` at this node's keys.
        void setValues(gtsam::Values &values, const ValuesType &node_values) const
        {
            setValuesBase_(values, node_values);
        }

    private:
        /// Insert typed node values at this node's keys; throws for a non-WNOA NodeValues.
        void setValuesBase_(gtsam::Values &values, const Types::NodeValues &node_values) const
        {
            const auto *typed_values = dynamic_cast<const ValuesType *>(&node_values);
            if (!typed_values)
                throw std::runtime_error("WNOA::Node::setValues received incompatible node values type.");
            this->insert("pose", typed_values->pose(), values);
            this->insert("strain", typed_values->epsilon(), values);
            this->insert("velocity", typed_values->varpi(), values);
        }

        gtsam::Key poseKey_, strainKey_, velocityKey_;
    };

}

// Node::interpolatedKeys needs InterpolatedKeys complete. It includes this file, so it can
// only come after Node is defined; its include guard breaks the cycle.
#include <spacetime/estimator/prior_models/WNOA/InterpolatedKeys.hpp>

#endif // WNOASTATE_HPP