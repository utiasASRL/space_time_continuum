#ifndef WNOV_COVARIANCES_HPP
#define WNOV_COVARIANCES_HPP

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <gtsam/base/Lie.h>

#include <spacetime/estimator/prior_models/CovarianceProfile.hpp>

namespace Spacetime::Priors::WNOV
{
    /**
     * @brief Arclength-dependent overrides of the WNOV prior's random-walk PSDs, given to
     *        WNOV::Prior at construction (see Priors::CovarianceOverride for the keying).
     *
     * Following WNOV::Prior's numbering, Q1 is the arclength-axis PSD, keyed by a segment's
     * lower node (`overrideQ1(0.0, Q)` sets the segment between 0 and the next node), and
     * Q2 the time-axis PSD, keyed by a node's arclength. Q0 (the base node's unary factor)
     * is not overridable.
     * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
     */
    template <typename Pose>
    struct CovarianceOverrides
    {
        static constexpr int dim = gtsam::traits<Pose>::dimension;
        using MatrixN = Eigen::Matrix<double, dim, dim>;
        using Override = CovarianceOverride<MatrixN>;

        Override Q1; ///< Arclength random-walk PSD, per segment.
        Override Q2; ///< Time random-walk PSD, per node.

        /// Q1 on the segment starting at the node at `arclength` (-1 for the last segment).
        CovarianceOverrides &overrideQ1(double arclength, const MatrixN &Q) { Q1.set(arclength, Q); return *this; }
        /// Q2 at the node at `arclength` (-1 for the tip).
        CovarianceOverrides &overrideQ2(double arclength, const MatrixN &Q) { Q2.set(arclength, Q); return *this; }
        /// Q1 on every segment, from its lower arclength and the nominal Q1.
        CovarianceOverrides &overrideQ1(typename Override::Function f) { Q1.set(std::move(f)); return *this; }
        /// Q2 at every node, from its arclength and the nominal Q2.
        CovarianceOverrides &overrideQ2(typename Override::Function f) { Q2.set(std::move(f)); return *this; }

        /// True if nothing is overridden.
        bool empty() const { return Q1.empty() && Q2.empty(); }
    };

    /**
     * @brief The WNOV random-walk PSDs resolved over a state's nodes and segments, shared by
     *        WNOV::Prior's factors and its Interpolator.
     * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
     */
    template <typename Pose>
    struct CovarianceProfiles
    {
        static constexpr int dim = gtsam::traits<Pose>::dimension;
        using MatrixN = Eigen::Matrix<double, dim, dim>;
        using Profile = CovarianceProfile<MatrixN>;

        Profile Q1; ///< Arclength random-walk PSD, per segment.
        Profile Q2; ///< Time random-walk PSD, per node.

        /// Constant PSDs.
        CovarianceProfiles(const MatrixN &Q1, const MatrixN &Q2)
            : Q1(Q1, CovarianceAxis::Space), Q2(Q2, CovarianceAxis::Time) {}

        /// The nominal PSDs with `overrides` resolved over the node `arclengths` (see Priors::CovarianceProfile).
        CovarianceProfiles(const MatrixN &Q1, const MatrixN &Q2, const std::vector<double> &arclengths, const CovarianceOverrides<Pose> &overrides)
            : Q1(Q1, CovarianceAxis::Space, arclengths, overrides.Q1, "WNOV::Prior Q1"),
              Q2(Q2, CovarianceAxis::Time, arclengths, overrides.Q2, "WNOV::Prior Q2") {}
    };
} // namespace Spacetime::Priors::WNOV

#endif // WNOV_COVARIANCES_HPP
