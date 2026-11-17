#ifndef WNOAQUASISTATIC_INTERPOLATEDKEYS_HPP
#define WNOAQUASISTATIC_INTERPOLATEDKEYS_HPP

#include <stdexcept>
#include <utility>

#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/prior_models/WNOAQuasiStatic/WNOAQuasiStaticState.hpp>

namespace Spacetime::Priors::WNOAQuasiStatic
{
    /**
     * @class InterpolatedKeys
     * @brief Well-known keys standing in for an interpolated quasi-static node's pose and
     *        strain (see Types::InterpolatedKeyMap for the generic interface, and
     *        WNOA::InterpolatedKeys for the full WNOA counterpart, whose Pose() and
     *        Strain() keys these mirror).
     *
     * There is no Velocity(), so sensors that need one (e.g. Sensors::GyroSensor) cannot
     * be built on this prior.
     */
    template <typename PoseT>
    struct InterpolatedKeys
    {
        using ValuesType = Values<PoseT>; ///< The node values these keys index.
        /// Dimension of each component (pose and strain both have the pose's tangent dimension).
        static constexpr int dim = gtsam::traits<PoseT>::dimension;

        /// Interpolated pose key.
        static gtsam::Key Pose() { return gtsam::Symbol('q', 0); }
        /// Interpolated strain key.
        static gtsam::Key Strain() { return gtsam::Symbol('q', 1); }

        /// The full set of interpolated keys this node type defines.
        static gtsam::KeyVector keys() { return {Pose(), Strain()}; }

        /// Insert the piece of `node` that `key` stands in for into `values`, at `key`; throws std::runtime_error for any other key.
        static void insert(gtsam::Key key, const ValuesType &node, gtsam::Values &values)
        {
            if (key == Pose())
                values.insert(key, node.pose());
            else if (key == Strain())
                values.insert(key, node.epsilon());
            else
                throw std::runtime_error("WNOAQuasiStatic::InterpolatedKeys::insert: key is not one of this node type's interpolated keys.");
        }

        /// The (offset, dim) row range `key`'s component occupies within the node's tangent vector; throws std::runtime_error for any other key.
        static std::pair<int, int> rowRange(gtsam::Key key)
        {
            if (key == Pose())
                return {0, dim};
            if (key == Strain())
                return {dim, dim};
            throw std::runtime_error("WNOAQuasiStatic::InterpolatedKeys::rowRange: key is not one of this node type's interpolated keys.");
        }
    };
} // namespace Spacetime::Priors::WNOAQuasiStatic

#endif // WNOAQUASISTATIC_INTERPOLATEDKEYS_HPP
