#ifndef WNOA_COVARIANCES_HPP
#define WNOA_COVARIANCES_HPP

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <gtsam/base/Lie.h>

#include <spacetime/estimator/prior_models/CovarianceProfile.hpp>

namespace Spacetime::Priors::WNOA
{
    /**
     * @brief Arclength-dependent overrides of the WNOA prior's process PSDs, given to
     *        WNOA::Prior at construction (see Priors::CovarianceOverride for the keying).
     *
     * Q1 and Q3 are time-axis covariances, keyed by a node's arclength. Q2 and Q4 are
     * arclength-axis covariances, keyed by a segment's lower node: `overrideQ2(0.0, Q)` sets
     * the segment between 0 and the next node. Q0 (the base node's unary factor) is not
     * overridable.
     * @code
     * WNOA::CovarianceOverrides<gtsam::Pose3> overrides;
     * overrides.overrideQ1(0.0, 10 * Q1)                    // the base node's time transitions
     *          .overrideQ2([](double s, const auto &Q) { return (1 + s) * Q; });
     * overrides.Q4.scale(-1, 0.5);                          // the last segment
     * @endcode
     * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
     */
    template <typename Pose>
    struct CovarianceOverrides
    {
        static constexpr int dim = gtsam::traits<Pose>::dimension;
        using MatrixN = Eigen::Matrix<double, dim, dim>;
        using Override = CovarianceOverride<MatrixN>;

        Override Q1; ///< Time-axis (pose, velocity) PSD, per node.
        Override Q2; ///< Arclength-axis (pose, strain) PSD, per segment.
        Override Q3; ///< Strain random-walk-in-time PSD, per node.
        Override Q4; ///< Velocity random-walk-in-arclength PSD, per segment.

        /// Q1 at the node at `arclength` (-1 for the tip).
        CovarianceOverrides &overrideQ1(double arclength, const MatrixN &Q) { Q1.set(arclength, Q); return *this; }
        /// Q2 on the segment starting at the node at `arclength` (-1 for the last segment).
        CovarianceOverrides &overrideQ2(double arclength, const MatrixN &Q) { Q2.set(arclength, Q); return *this; }
        /// Q3 at the node at `arclength` (-1 for the tip).
        CovarianceOverrides &overrideQ3(double arclength, const MatrixN &Q) { Q3.set(arclength, Q); return *this; }
        /// Q4 on the segment starting at the node at `arclength` (-1 for the last segment).
        CovarianceOverrides &overrideQ4(double arclength, const MatrixN &Q) { Q4.set(arclength, Q); return *this; }
        /// Q1 at every node, from its arclength and the nominal Q1.
        CovarianceOverrides &overrideQ1(typename Override::Function f) { Q1.set(std::move(f)); return *this; }
        /// Q2 on every segment, from its lower arclength and the nominal Q2.
        CovarianceOverrides &overrideQ2(typename Override::Function f) { Q2.set(std::move(f)); return *this; }
        /// Q3 at every node, from its arclength and the nominal Q3.
        CovarianceOverrides &overrideQ3(typename Override::Function f) { Q3.set(std::move(f)); return *this; }
        /// Q4 on every segment, from its lower arclength and the nominal Q4.
        CovarianceOverrides &overrideQ4(typename Override::Function f) { Q4.set(std::move(f)); return *this; }

        /// True if nothing is overridden.
        bool empty() const { return Q1.empty() && Q2.empty() && Q3.empty() && Q4.empty(); }
    };

    /**
     * @brief The WNOA process PSDs resolved over a state's nodes and segments, shared by
     *        WNOA::Prior's factors and its Interpolator.
     * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
     */
    template <typename Pose>
    struct CovarianceProfiles
    {
        static constexpr int dim = gtsam::traits<Pose>::dimension;
        using MatrixN = Eigen::Matrix<double, dim, dim>;
        using Profile = CovarianceProfile<MatrixN>;

        Profile Q1; ///< Time-axis (pose, velocity) PSD, per node.
        Profile Q2; ///< Arclength-axis (pose, strain) PSD, per segment.
        Profile Q3; ///< Strain random-walk-in-time PSD, per node.
        Profile Q4; ///< Velocity random-walk-in-arclength PSD, per segment.

        /// Constant PSDs.
        CovarianceProfiles(const MatrixN &Q1, const MatrixN &Q2, const MatrixN &Q3, const MatrixN &Q4)
            : Q1(Q1, CovarianceAxis::Time), Q2(Q2, CovarianceAxis::Space), Q3(Q3, CovarianceAxis::Time), Q4(Q4, CovarianceAxis::Space) {}

        /// Already resolved profiles.
        CovarianceProfiles(Profile Q1, Profile Q2, Profile Q3, Profile Q4)
            : Q1(std::move(Q1)), Q2(std::move(Q2)), Q3(std::move(Q3)), Q4(std::move(Q4)) {}

        /// The nominal PSDs with `overrides` resolved over the node `arclengths` (see Priors::CovarianceProfile).
        CovarianceProfiles(const MatrixN &Q1, const MatrixN &Q2, const MatrixN &Q3, const MatrixN &Q4,
                           const std::vector<double> &arclengths, const CovarianceOverrides<Pose> &overrides)
            : Q1(Q1, CovarianceAxis::Time, arclengths, overrides.Q1, "WNOA::Prior Q1"),
              Q2(Q2, CovarianceAxis::Space, arclengths, overrides.Q2, "WNOA::Prior Q2"),
              Q3(Q3, CovarianceAxis::Time, arclengths, overrides.Q3, "WNOA::Prior Q3"),
              Q4(Q4, CovarianceAxis::Space, arclengths, overrides.Q4, "WNOA::Prior Q4") {}
    };
} // namespace Spacetime::Priors::WNOA

#endif // WNOA_COVARIANCES_HPP
