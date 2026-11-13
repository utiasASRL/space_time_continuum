/**
 * @file Ellipsoid.hpp
 * @brief Confidence ellipsoids of 3D Gaussians, e.g. a position's posterior covariance,
 *        shared by every way of drawing one (visualizer backends, ROS2 markers).
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_UTILITIES_ELLIPSOID_HPP
#define SPACETIME_UTILITIES_ELLIPSOID_HPP

#include <algorithm>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace Spacetime::Utilities
{
    /// sqrt of the chi-squared 99.7% quantile with 3 degrees of freedom (13.9314): the
    /// Mahalanobis radius enclosing 99.7% of a 3D Gaussian.
    constexpr double kEllipsoidScale99_7 = 3.7324821051831023;

    /**
     * @brief Transform mapping a unit sphere onto the confidence ellipsoid of a 3D
     *        Gaussian: its columns are the covariance's eigenvectors (a proper rotation)
     *        scaled by `scale * sqrt(eigenvalue)`, and its translation is `position`.
     *
     * Eigenvalues are clamped to be non-negative before the square root (a posterior
     * covariance can be very slightly indefinite numerically), and each semi-axis is kept
     * at least 1e-8 so the transform stays invertible.
     */
    inline Eigen::Matrix4d ellipsoidTransform(const Eigen::Vector3d &position, const Eigen::Matrix3d &covariance, double scale = kEllipsoidScale99_7)
    {
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(0.5 * (covariance + covariance.transpose()));
        Eigen::Matrix3d axes = solver.eigenvectors();
        if (axes.determinant() < 0.0)
            axes.col(0) = -axes.col(0); // keep a proper rotation

        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        for (int i = 0; i < 3; i++)
        {
            const double semi_axis = std::max(scale * std::sqrt(std::max(solver.eigenvalues()(i), 0.0)), 1e-8);
            transform.block<3, 1>(0, i) = semi_axis * axes.col(i);
        }
        transform.block<3, 1>(0, 3) = position;
        return transform;
    }
} // namespace Spacetime::Utilities

#endif // SPACETIME_UTILITIES_ELLIPSOID_HPP
