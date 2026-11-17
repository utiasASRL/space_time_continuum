#ifndef WNOA_INTERPOLATEDKEYS_HPP
#define WNOA_INTERPOLATEDKEYS_HPP

#include <stdexcept>
#include <utility>

#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/prior_models/WNOA/WNOAState.hpp>

namespace Spacetime::Priors::WNOA
{
    /**
     * @class InterpolatedKeys
     * @brief WNOA-specific well-known keys standing in for an interpolated node's
     *        pose, strain, and velocity components.
     *
     * This is where the WNOA node type's own interpolated-key layout is defined
     * (see WNOA::Values<Pose>/WNOA::Node<Pose> in WNOAState.hpp) -- the generic
     * Priors::InterpolationFactor and Types::SensorBase never hardcode "pose",
     * "strain", or "velocity" themselves; they only require whatever type is named
     * by `InterpolatorT::InterpolatedKeys` / `ValuesT::InterpolatedKeys` to satisfy
     * Types::InterpolatedKeyMap (see spacetime/estimator/types/InterpolatedKeys.hpp
     * for that interface's full documentation). Each other prior defines its own
     * analogous type in its own folder (e.g. WNOV::InterpolatedKeys,
     * WNOAQuasiStatic::InterpolatedKeys), with its own keys/insert/rowRange.
     *
     * Named accessors (Pose()/Strain()/Velocity()) are provided in addition to the
     * generic interface for callers -- e.g. Sensors::PoseSensor -- that want a
     * specific, semantically-known WNOA key by name rather than iterating the
     * generic key set.
     *
     * These keys only need to be distinct from each other and from any
     * genuinely-outer key a sensor's factor also references directly -- NOT
     * globally unique across factor instances or over time -- because the
     * gtsam::Values used to evaluate a sensor's factor is built fresh, locally,
     * inside InterpolationFactor::computeInterpolatedError on every call; it is
     * never inserted into the graph's real Values.
     */
    template <typename PoseT>
    struct InterpolatedKeys
    {
        using ValuesType = Values<PoseT>; ///< The node values these keys index.
        /// Dimension of each component (pose, strain and velocity all have the pose's tangent dimension).
        static constexpr int dim = gtsam::traits<PoseT>::dimension;

        /// Interpolated pose key.
        static gtsam::Key Pose() { return gtsam::Symbol('q', 0); }
        /// Interpolated strain key.
        static gtsam::Key Strain() { return gtsam::Symbol('q', 1); }
        /// Interpolated velocity key.
        static gtsam::Key Velocity() { return gtsam::Symbol('q', 2); }

        /// The full set of interpolated keys this node type defines.
        static gtsam::KeyVector keys() { return {Pose(), Strain(), Velocity()}; }

        /// Insert the piece of `node` that `key` stands in for into `values`, at `key`; throws std::runtime_error for any other key.
        static void insert(gtsam::Key key, const ValuesType &node, gtsam::Values &values)
        {
            if (key == Pose())
                values.insert(key, node.pose());
            else if (key == Strain())
                values.insert(key, node.epsilon());
            else if (key == Velocity())
                values.insert(key, node.varpi());
            else
                throw std::runtime_error("WNOA::InterpolatedKeys::insert: key is not one of this node type's interpolated keys.");
        }

        /// The (offset, dim) row range `key`'s component occupies within the node's node_dim tangent vector; throws std::runtime_error for any other key.
        static std::pair<int, int> rowRange(gtsam::Key key)
        {
            if (key == Pose())
                return {0, dim};
            if (key == Strain())
                return {dim, dim};
            if (key == Velocity())
                return {2 * dim, dim};
            throw std::runtime_error("WNOA::InterpolatedKeys::rowRange: key is not one of this node type's interpolated keys.");
        }
    };
} // namespace Spacetime::Priors::WNOA

#endif // WNOA_INTERPOLATEDKEYS_HPP
