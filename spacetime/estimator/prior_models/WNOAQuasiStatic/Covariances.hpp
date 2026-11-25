#ifndef WNOAQUASISTATIC_COVARIANCES_HPP
#define WNOAQUASISTATIC_COVARIANCES_HPP

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <gtsam/base/Lie.h>

#include <spacetime/estimator/prior_models/CovarianceProfile.hpp>

namespace Spacetime::Priors::WNOAQuasiStatic
{
    /**
     * @brief Arclength-dependent overrides of the quasi-static prior's arclength-axis PSD
     *        Q2, given to WNOAQuasiStatic::Prior at construction (see
     *        Priors::CovarianceOverride). Q2 is keyed by a segment's lower node:
     *        `overrideQ2(0.0, Q)` sets the segment between 0 and the next node. Q0 is not
     *        overridable.
     * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
     */
    template <typename Pose>
    struct CovarianceOverrides
    {
        static constexpr int dim = gtsam::traits<Pose>::dimension;
        using MatrixN = Eigen::Matrix<double, dim, dim>;
        using Override = CovarianceOverride<MatrixN>;

        Override Q2; ///< Arclength-axis (pose, strain) PSD, per segment.

        /// Q2 on the segment starting at the node at `arclength` (-1 for the last segment).
        CovarianceOverrides &overrideQ2(double arclength, const MatrixN &Q) { Q2.set(arclength, Q); return *this; }
        /// Q2 on every segment, from its lower arclength and the nominal Q2.
        CovarianceOverrides &overrideQ2(typename Override::Function f) { Q2.set(std::move(f)); return *this; }

        /// True if nothing is overridden.
        bool empty() const { return Q2.empty(); }
    };

    /**
     * @brief The quasi-static prior's Q2 resolved over a state's segments, shared by
     *        WNOAQuasiStatic::Prior's factors and its Interpolator.
     * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
     */
    template <typename Pose>
    struct CovarianceProfiles
    {
        static constexpr int dim = gtsam::traits<Pose>::dimension;
        using MatrixN = Eigen::Matrix<double, dim, dim>;
        using Profile = CovarianceProfile<MatrixN>;

        Profile Q2; ///< Arclength-axis (pose, strain) PSD, per segment.

        /// A constant Q2.
        explicit CovarianceProfiles(const MatrixN &Q2) : Q2(Q2, CovarianceAxis::Space) {}

        /// The nominal Q2 with `overrides` resolved over the node `arclengths` (see Priors::CovarianceProfile).
        CovarianceProfiles(const MatrixN &Q2, const std::vector<double> &arclengths, const CovarianceOverrides<Pose> &overrides)
            : Q2(Q2, CovarianceAxis::Space, arclengths, overrides.Q2, "WNOAQuasiStatic::Prior Q2") {}
    };
} // namespace Spacetime::Priors::WNOAQuasiStatic

#endif // WNOAQUASISTATIC_COVARIANCES_HPP
