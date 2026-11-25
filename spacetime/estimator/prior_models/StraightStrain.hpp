#ifndef STRAIGHTSTRAIN_HPP
#define STRAIGHTSTRAIN_HPP

#include <algorithm>
#include <cmath>
#include <string>
#include <type_traits>

#include <gtsam/base/Lie.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>

#include <spacetime/estimator/utilities/logger.hpp>

namespace Spacetime::Priors
{
    /**
     * @brief The library's tangent-axis convention: the backbone runs along each node's body x
     *        axis, so a straight, unstretched robot has a unit translational strain along x and
     *        no rotational strain ([0, 0, 0, 1, 0, 0] for gtsam::Pose3 in GTSAM's
     *        [rotational, translational] order; [1, 0, 0] for gtsam::Pose2, whose tangent is
     *        [x, y, theta]).
     * @tparam Pose gtsam::Pose3 or gtsam::Pose2.
     */
    template <typename Pose>
    typename gtsam::traits<Pose>::TangentVector straightStrain()
    {
        typename gtsam::traits<Pose>::TangentVector epsilon = gtsam::traits<Pose>::TangentVector::Zero();
        if constexpr (std::is_same_v<Pose, gtsam::Pose3>)
            epsilon(3) = 1.0;
        else if constexpr (std::is_same_v<Pose, gtsam::Pose2>)
            epsilon(0) = 1.0;
        else
            static_assert(std::is_same_v<Pose, gtsam::Pose3> || std::is_same_v<Pose, gtsam::Pose2>,
                          "straightStrain supports gtsam::Pose3 and gtsam::Pose2.");
        return epsilon;
    }

    /**
     * @brief Whether `epsilon`'s translational strain has the body x axis as its tangent axis:
     *        its x component is positive and at least as large as the other translational
     *        components (see straightStrain).
     * @tparam Pose gtsam::Pose3 or gtsam::Pose2.
     */
    template <typename Pose>
    bool isTangentX(const typename gtsam::traits<Pose>::TangentVector &epsilon)
    {
        if constexpr (std::is_same_v<Pose, gtsam::Pose3>)
            return epsilon(3) > 0.0 && epsilon(3) >= std::max(std::abs(epsilon(4)), std::abs(epsilon(5)));
        else
            return epsilon(0) > 0.0 && epsilon(0) >= std::abs(epsilon(1));
    }

    /**
     * @brief Warns when a prior's nominal strain `epsilon` does not have the body x axis as
     *        its tangent axis (see isTangentX): the library's sensors, actuators and
     *        visualizer assume it does.
     * @param prior_name The prior named in the warning (e.g. "WNOA::Prior").
     * @return Whether it warned.
     */
    template <typename Pose>
    bool warnIfTangentNotX(const typename gtsam::traits<Pose>::TangentVector &epsilon, const std::string &prior_name)
    {
        if (isTangentX<Pose>(epsilon))
            return false;
        warning() << prior_name << ": the nominal strain [" << epsilon.transpose()
                  << "] does not have the body x axis as its tangent axis. The library assumes the backbone runs along x, "
                     "e.g. a straight robot has strain ["
                  << straightStrain<Pose>().transpose() << "] (GTSAM [rotational, translational] order).";
        return true;
    }
} // namespace Spacetime::Priors

#endif // STRAIGHTSTRAIN_HPP
