#ifndef INTERPOLATEDKEYS_HPP
#define INTERPOLATEDKEYS_HPP

#include <concepts>
#include <utility>

#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/Values.h>

namespace Spacetime::Types
{
    /**
     * @brief Generic interface a prior-specific "interpolated key map" type must
     *        satisfy for Priors::InterpolationFactor to remain agnostic to what an
     *        interpolated node's sub-state components actually are.
     *
     * An interpolated node (see e.g. Priors::GridInterpolator::interpolate) is, in
     * general, made up of some number of independently-keyed sub-states (WNOA's
     * happens to be pose/strain/velocity, but a different prior's node could be
     * made up of any other set). A sensor builds its measurement factor directly
     * on whichever of a node type's well-known "interpolated" keys it needs (see
     * Types::SensorBase::getFactor), and Priors::InterpolationFactor evaluates
     * that factor against the actual interpolated node -- mirroring
     * gtsam::WnoaInterpFactor's convention of the caller building the inner
     * factor on caller-chosen "interpolated" keys, generalized from gtsam's fixed
     * (pose, velocity) pair to an arbitrary, prior-defined set of sub-states.
     *
     * Concrete implementations live alongside their node type's own definition
     * (e.g. WNOA::InterpolatedKeys<Pose>, in
     * spacetime/estimator/prior_models/WNOA/InterpolatedKeys.hpp) -- this file
     * only defines the shape such a type must have, never the keys themselves.
     * `ValuesT` is the prior's own node-values type (e.g. WNOA::Values<Pose>).
     *
     * A conforming `T`:
     *   - `T::keys()` returns the full set of well-known "interpolated" keys a
     *     sensor's inner factor may reference for this node type.
     *   - `T::insert(key, node, values)` inserts the piece of `node` that `key`
     *     stands in for into `values`, at `key` (type-erased via gtsam::Values's
     *     own type erasure, since different sub-states are generally different
     *     C++ types).
     *   - `T::rowRange(key)` returns the (offset, dim) row range that `key`'s
     *     component occupies within the node's full node_dim tangent vector, used
     *     to slice the interpolator's per-corner-key Jacobian blocks and the
     *     conditional covariance without InterpolationFactor needing to know the
     *     node's internal layout.
     *
     * These only need to be distinct from each other and from any genuinely-outer
     * key a sensor's factor also references directly (e.g. a landmark key) -- NOT
     * globally unique across factor instances or over time -- because the
     * gtsam::Values used to evaluate a sensor's factor is built fresh, locally,
     * inside InterpolationFactor::computeInterpolatedError on every call; it is
     * never inserted into the graph's real Values.
     */
    template <typename T, typename ValuesT>
    concept InterpolatedKeyMap = requires(gtsam::Key key, const ValuesT &node, gtsam::Values &values) {
        { T::keys() } -> std::convertible_to<gtsam::KeyVector>;
        { T::insert(key, node, values) };
        { T::rowRange(key) } -> std::convertible_to<std::pair<int, int>>;
    };
} // namespace Spacetime::Types

#endif // INTERPOLATEDKEYS_HPP
