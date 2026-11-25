#ifndef INTERPOLATEDKEYS_HPP
#define INTERPOLATEDKEYS_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
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
     * `ValuesT` is the prior's own node-values type (e.g. WNOA::NodeValues<Pose>).
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

    /**
     * Point-indexed interpolated keys: a factor spanning several arclengths at once (e.g. a
     * tendon over its routing disks, see Types::ActuatorBase::getFactors) references
     * interpolated key K at the i-th of those arclengths as InterpolatedKeyAt(K, i), which
     * every node type's keys expose as an overload (e.g. WNOA::InterpolatedKeys::Pose(i)).
     * The plain keys (Pose()) stay the single query point of a sensor's factor.
     *
     * The encoding, Symbol(K.chr(), POINT_FLAG | i << 32 | K.index()), is shared by every node
     * type, so a point key is told apart from a genuine variable without knowing the node
     * type (KeyManager's counters never reach POINT_FLAG).
     */
    inline constexpr std::uint64_t POINT_FLAG = std::uint64_t(1) << 55;
    /// Largest point index InterpolatedKeyAt accepts.
    inline constexpr std::size_t MAX_POINT = (std::size_t(1) << 23) - 1;

    /// Whether `key` is a point-indexed interpolated key (see InterpolatedKeyAt).
    inline bool isPointKey(gtsam::Key key) { return (gtsam::Symbol(key).index() & POINT_FLAG) != 0; }

    /**
     * @brief Interpolated key `placeholder` (e.g. WNOA::InterpolatedKeys::Pose()) at the
     *        `point`-th arclength of a multi-point factor.
     * @throws std::invalid_argument if `placeholder` is already point-indexed or its index
     *         does not fit in 32 bits, or `point` exceeds MAX_POINT.
     */
    inline gtsam::Key InterpolatedKeyAt(gtsam::Key placeholder, std::size_t point)
    {
        const gtsam::Symbol symbol(placeholder);
        if (isPointKey(placeholder) || symbol.index() >> 32 != 0)
            throw std::invalid_argument("InterpolatedKeyAt: the placeholder must be a plain interpolated key.");
        if (point > MAX_POINT)
            throw std::invalid_argument("InterpolatedKeyAt: the point index is too large.");
        return gtsam::Symbol(symbol.chr(), POINT_FLAG | (std::uint64_t(point) << 32) | symbol.index());
    }

    /// The point index of a point-indexed key (see InterpolatedKeyAt); nullopt for any other key.
    inline std::optional<std::size_t> pointOf(gtsam::Key key)
    {
        if (!isPointKey(key))
            return std::nullopt;
        return static_cast<std::size_t>((gtsam::Symbol(key).index() & ~POINT_FLAG) >> 32);
    }

    /// The plain interpolated key a point-indexed key stands for (see InterpolatedKeyAt); any other key unchanged.
    inline gtsam::Key placeholderOf(gtsam::Key key)
    {
        if (!isPointKey(key))
            return key;
        const gtsam::Symbol symbol(key);
        return gtsam::Symbol(symbol.chr(), symbol.index() & 0xFFFFFFFFu);
    }

    /// The sorted, distinct point indices of `factor`'s point-indexed keys.
    inline std::vector<std::size_t> pointsOf(const gtsam::NonlinearFactor &factor)
    {
        std::vector<std::size_t> points;
        for (const gtsam::Key key : factor.keys())
            if (const auto point = pointOf(key))
                points.push_back(*point);
        std::sort(points.begin(), points.end());
        points.erase(std::unique(points.begin(), points.end()), points.end());
        return points;
    }

    /// pointsOf(factor). @throws std::out_of_range if a point index is not below `count`.
    inline std::vector<std::size_t> pointsOf(const gtsam::NonlinearFactor &factor, std::size_t count)
    {
        std::vector<std::size_t> points = pointsOf(factor);
        if (!points.empty() && points.back() >= count)
            throw std::out_of_range("The factor references point " + std::to_string(points.back()) + ", but only " +
                                    std::to_string(count) + " arclengths were given.");
        return points;
    }

    /// `keys`, each as InterpolatedKeyAt(key, point).
    inline gtsam::KeyVector InterpolatedKeysAt(const gtsam::KeyVector &keys, std::size_t point)
    {
        gtsam::KeyVector indexed;
        for (const gtsam::Key key : keys)
            indexed.push_back(InterpolatedKeyAt(key, point));
        return indexed;
    }
} // namespace Spacetime::Types

#endif // INTERPOLATEDKEYS_HPP
