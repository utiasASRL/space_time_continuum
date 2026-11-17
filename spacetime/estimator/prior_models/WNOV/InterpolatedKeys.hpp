#ifndef WNOV_INTERPOLATEDKEYS_HPP
#define WNOV_INTERPOLATEDKEYS_HPP

#include <stdexcept>
#include <utility>

#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/prior_models/WNOV/WNOVState.hpp>

namespace Spacetime::Priors::WNOV
{
    /**
     * @class InterpolatedKeys
     * @brief WNOV-specific well-known key standing in for an interpolated node's pose,
     *        the only component of a WNOV node (see Types::InterpolatedKeyMap for the
     *        generic interface, and WNOA::InterpolatedKeys for its WNOA counterpart).
     *
     * Pose() is provided by name for callers -- e.g. Sensors::PoseSensor -- that want the
     * pose key specifically. Like every interpolated key it only needs to be distinct from
     * any genuinely-outer key a sensor's factor references, since it is only ever used in
     * the local gtsam::Values built inside InterpolationFactor::computeInterpolatedError.
     */
    template <typename PoseT>
    struct InterpolatedKeys
    {
        using ValuesType = Values<PoseT>; ///< The node values these keys index.
        /// Pose tangent dimension.
        static constexpr int dim = gtsam::traits<PoseT>::dimension;

        /// Interpolated pose key.
        static gtsam::Key Pose() { return gtsam::Symbol('q', 0); }

        /// The full set of interpolated keys this node type defines.
        static gtsam::KeyVector keys() { return {Pose()}; }

        /// Insert the pose of `node` into `values` at `key`; throws std::runtime_error unless `key` is Pose().
        static void insert(gtsam::Key key, const ValuesType &node, gtsam::Values &values)
        {
            if (key != Pose())
                throw std::runtime_error("WNOV::InterpolatedKeys::insert: key is not one of this node type's interpolated keys.");
            values.insert(key, node.pose());
        }

        /// The (offset, dim) row range `key`'s component occupies within the node's tangent vector ({0, dim}); throws std::runtime_error unless `key` is Pose().
        static std::pair<int, int> rowRange(gtsam::Key key)
        {
            if (key != Pose())
                throw std::runtime_error("WNOV::InterpolatedKeys::rowRange: key is not one of this node type's interpolated keys.");
            return {0, dim};
        }
    };
} // namespace Spacetime::Priors::WNOV

#endif // WNOV_INTERPOLATEDKEYS_HPP
