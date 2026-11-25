#ifndef KINEMATICTENDON_HPP
#define KINEMATICTENDON_HPP

#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/prior_models/MultiInterpolationFactor.hpp>
#include <spacetime/estimator/types/ActuatorBase.hpp>
#include <spacetime/estimator/types/Measurement.hpp>
#include <spacetime/estimator/utilities/logger.hpp>

namespace Spacetime::Actuators
{
    /**
     * @class TendonLengthFactor
     * @brief The kinematic tendon-length residual: e = L(T_0, ..., T_{N-1}) - u, where
     *        L = sum_i |p_{i+1} - p_i| is the length of the straight-line path through the
     *        routing points p_i = T_i * offset (the tendon's offset from the backbone,
     *        expressed in each routing disk's body frame), and u the tendon-length variable
     *        (a 1-vector).
     */
    class TendonLengthFactor : public gtsam::NoiseModelFactor
    {
    public:
        using Base = gtsam::NoiseModelFactor;

        /**
         * @param pose_keys The routing disks' pose keys, base to tip (at least two).
         * @param length_key The tendon-length variable's key (a gtsam::Vector of size 1).
         * @param offset The routing points' offset in each disk's body frame [m].
         * @param noise_model A 1-dimensional noise model.
         * @throws std::invalid_argument for fewer than two pose keys or a noise model that is not 1-dimensional.
         */
        TendonLengthFactor(const gtsam::KeyVector &pose_keys, gtsam::Key length_key, const gtsam::Point3 &offset,
                           const gtsam::SharedNoiseModel &noise_model)
            : Base(noise_model, withLength(pose_keys, length_key)), m_num_disks(pose_keys.size()), m_offset(offset)
        {
            if (pose_keys.size() < 2)
                throw std::invalid_argument("TendonLengthFactor needs at least two routing disks.");
            if (!noise_model || noise_model->dim() != 1)
                throw std::invalid_argument("TendonLengthFactor needs a 1-dimensional noise model.");
        }

        /// The routing points' offset in each disk's body frame [m].
        const gtsam::Point3 &offset() const { return m_offset; }

        /**
         * @brief The length of the straight-line path through `offset` carried by every pose in `poses`.
         * @param H Optional output: one 1x6 Jacobian per pose.
         */
        static double pathLength(const std::vector<gtsam::Pose3> &poses, const gtsam::Point3 &offset, std::vector<gtsam::Matrix> *H = nullptr)
        {
            const size_t n = poses.size();
            std::vector<gtsam::Point3> points(n);
            std::vector<gtsam::Matrix36> H_points(n);
            for (size_t i = 0; i < n; ++i)
                points[i] = poses[i].transformFrom(offset, H ? &H_points[i] : nullptr);
            if (H)
                H->assign(n, gtsam::Matrix::Zero(1, 6));

            double length = 0.0;
            for (size_t i = 0; i + 1 < n; ++i)
            {
                const gtsam::Vector3 d = points[i + 1] - points[i];
                const double segment = d.norm();
                length += segment;
                if (H && segment > std::numeric_limits<double>::epsilon())
                {
                    const Eigen::RowVector3d direction = d.transpose() / segment;
                    (*H)[i + 1] += direction * H_points[i + 1];
                    (*H)[i] -= direction * H_points[i];
                }
            }
            return length;
        }

        using Base::unwhitenedError;
        /// L - u, with Jacobians w.r.t. every pose and the tendon length.
        gtsam::Vector unwhitenedError(const gtsam::Values &values, gtsam::OptionalMatrixVecType H = nullptr) const override
        {
            std::vector<gtsam::Pose3> poses(m_num_disks);
            for (size_t i = 0; i < m_num_disks; ++i)
                poses[i] = values.at<gtsam::Pose3>(keys_[i]);
            const gtsam::Vector u = values.at<gtsam::Vector>(keys_.back());
            if (u.size() != 1)
                throw std::invalid_argument("TendonLengthFactor: the tendon length must be a vector of size 1.");

            std::vector<gtsam::Matrix> H_poses;
            const double length = pathLength(poses, m_offset, H ? &H_poses : nullptr);
            if (H)
            {
                H->resize(keys_.size());
                for (size_t i = 0; i < m_num_disks; ++i)
                    (*H)[i] = H_poses[i];
                (*H)[m_num_disks] = -gtsam::Matrix::Identity(1, 1);
            }
            return gtsam::Vector1(length - u(0));
        }

        /// Print the keys and the offset.
        void print(const std::string &s = "", const gtsam::KeyFormatter &keyFormatter = gtsam::DefaultKeyFormatter) const override
        {
            std::cout << s << "TendonLengthFactor on ";
            for (const auto &k : keys())
                std::cout << keyFormatter(k) << " ";
            std::cout << "with offset " << m_offset.transpose() << std::endl;
            if (noiseModel())
                noiseModel()->print("  noise model: ");
        }

        /// Same type, keys, offset and noise model.
        bool equals(const gtsam::NonlinearFactor &expected, double tol = 1e-9) const override
        {
            const auto *e = dynamic_cast<const TendonLengthFactor *>(&expected);
            return e != nullptr && Base::equals(*e, tol) && gtsam::equal_with_abs_tol(m_offset, e->m_offset, tol);
        }

    private:
        static gtsam::KeyVector withLength(gtsam::KeyVector keys, gtsam::Key length_key)
        {
            keys.push_back(length_key);
            return keys;
        }

        size_t m_num_disks;
        gtsam::Point3 m_offset;
    };

    /**
     * @class KinematicTendon
     * @brief A single tendon routed along the robot through evenly spaced disks: an actuator
     *        (see Types::ActuatorBase) whose input is the measured tendon length.
     *
     *        Each measurement adds a tendon-length variable u_k at its time, tied to the
     *        measured length by its covariance. One factor per u_k compares it with the
     *        length of the straight-line path through the routing points traced by the
     *        current state estimate (see TendonLengthFactor). The routing points lie at
     *        arclengths 0, d, 2d, ... up to `termination` (always the last disk), where d is
     *        `discretization`, or at arclengths given explicitly; each sits `radius` from
     *        the backbone at `angle` about it.
     *        The backbone runs along each node's body x axis (the straight strain is
     *        [0, 0, 0, 1, 0, 0]), so the offset in the body frame is
     *        [0, radius cos(angle), radius sin(angle)].
     *
     *        The disks may lie between nodes: the prior interpolates every disk's pose
     *        jointly (see Priors::GridPrior::getMultiPointOuterFactor). A measurement's
     *        arclength is unused.
     *        The factor's own noise, lengthSigma(), is tight by default, so the measurement's
     *        covariance (on u_k) sets how much the tendon length is trusted.
     *
     * @tparam PriorT The robot's prior (a Priors::GridPrior with gtsam::Pose3 poses, e.g.
     *         Priors::WNOA::Prior<gtsam::Pose3>).
     */
    template <typename PriorT>
    class KinematicTendon : public Types::ActuatorBase
    {
    public:
        using PriorType = PriorT;                                              ///< The robot's prior.
        using InterpolatorType = typename PriorT::InterpolatorType;            ///< The prior's interpolator.
        using Keys = typename InterpolatorType::InterpolatedKeys;              ///< Interpolated keys of the robot's nodes.
        using FactorType = Priors::MultiInterpolationFactor<InterpolatorType>; ///< The outer factor of every input (built by the prior).

        static_assert(std::is_same_v<std::decay_t<decltype(std::declval<typename PriorT::ValuesType>().pose())>, gtsam::Pose3>,
                      "KinematicTendon requires the robot's poses to be gtsam::Pose3.");

        /// Default standard deviation of the tendon-length factor [m]: tight, yet loose enough
        /// to keep the problem well conditioned (at 1e-6, Levenberg-Marquardt stalls).
        static constexpr double DEFAULT_LENGTH_SIGMA = 1e-4;

        /**
         * @param prior The prior of the robot the tendon is routed along.
         * @param radius Distance of the routing points from the backbone [m], >= 0.
         * @param angle Angle of the routing points about the backbone [rad], from the body y axis towards z.
         * @param termination Arclength where the tendon ends [m], in (0, robot length].
         * @param discretization Spacing of the routing disks along the backbone [m], > 0.
         * @throws std::runtime_error for a null prior; std::invalid_argument for an invalid
         *         radius, termination or discretization.
         */
        KinematicTendon(std::shared_ptr<PriorT> prior, double radius, double angle, double termination, double discretization)
            : KinematicTendon(prior, radius, angle, evenDisks(prior, termination, discretization), discretization)
        {
        }

        /**
         * @param prior The prior of the robot the tendon is routed along.
         * @param radius Distance of the routing points from the backbone [m], >= 0.
         * @param angle Angle of the routing points about the backbone [rad], from the body y axis towards z.
         * @param disk_arclengths The routing disks' arclengths [m]: at least two, strictly
         *        ascending, within [0, robot length]. The tendon ends at the last one.
         * @throws std::runtime_error for a null prior; std::invalid_argument for an invalid
         *         radius or disk arclengths.
         */
        KinematicTendon(std::shared_ptr<PriorT> prior, double radius, double angle, std::vector<double> disk_arclengths)
            : KinematicTendon(prior, radius, angle, std::move(disk_arclengths), std::nullopt)
        {
        }

    private:
        KinematicTendon(std::shared_ptr<PriorT> prior, double radius, double angle, std::vector<double> disk_arclengths,
                        std::optional<double> discretization)
            : Types::ActuatorBase(prior, checkedDisks(prior, radius, angle, std::move(disk_arclengths)), 1),
              mp_prior(std::move(prior)), m_radius(radius), m_angle(angle), m_termination(arclengths().back()),
              m_discretization(discretization),
              m_offset(0.0, radius * std::cos(angle), radius * std::sin(angle))
        {
        }

    public:
        /**
         * @brief The routing disks' arclengths: 0, d, 2d, ... while below `termination`
         *        (within TOLERANCE), then `termination` itself.
         * @throws std::invalid_argument for a non-positive termination or discretization.
         */
        static std::vector<double> diskArclengths(double termination, double discretization)
        {
            if (!(discretization > 0.0) || !std::isfinite(discretization))
                throw std::invalid_argument("KinematicTendon: the discretization must be positive.");
            if (!(termination > 0.0) || !std::isfinite(termination))
                throw std::invalid_argument("KinematicTendon: the termination must be positive.");
            std::vector<double> disks;
            for (unsigned int k = 0; k * discretization < termination - TOLERANCE; ++k)
                disks.push_back(k * discretization);
            disks.push_back(termination);
            return disks;
        }

        /// The robot's prior.
        const std::shared_ptr<PriorT> &prior() const { return mp_prior; }
        /// Distance of the routing points from the backbone [m].
        double radius() const { return m_radius; }
        /// Angle of the routing points about the backbone [rad].
        double angle() const { return m_angle; }
        /// Arclength where the tendon ends [m].
        double termination() const { return m_termination; }
        /// Spacing of the routing disks [m]; nullopt when they were given explicitly.
        std::optional<double> discretization() const { return m_discretization; }
        /// The routing points' offset in each disk's body frame [m].
        const gtsam::Point3 &offset() const { return m_offset; }

        /// Standard deviation of the tendon-length factor [m] (see the class description).
        double lengthSigma() const { return m_length_sigma; }
        /// Set the standard deviation of the tendon-length factor [m]. @throws std::invalid_argument if not positive.
        void setLengthSigma(double sigma)
        {
            if (!(sigma > 0.0))
                throw std::invalid_argument("KinematicTendon: the length sigma must be positive.");
            m_length_sigma = sigma;
        }

        /// The tendon length predicted by `poses`, one per routing disk (see TendonLengthFactor::pathLength).
        double predictedLength(const std::vector<gtsam::Pose3> &poses) const
        {
            return TendonLengthFactor::pathLength(poses, m_offset);
        }

        /// The inner key standing in for routing disk `disk`'s pose in getFactors' factor (Keys::Pose(disk)).
        static gtsam::Key diskPoseKey(size_t disk) { return Keys::Pose(disk); }

        /**
         * @brief The tendon-length factor of the input `input_key`: one factor on every
         *        routing disk's interpolated pose (diskPoseKey) and the tendon length.
         */
        InnerFactors getFactors(const Types::MeasurementBase &measurement, gtsam::Key input_key) const override
        {
            (void)measurement;
            gtsam::KeyVector pose_keys;
            for (size_t i = 0; i < arclengths().size(); ++i)
                pose_keys.push_back(diskPoseKey(i));
            return {std::make_shared<TendonLengthFactor>(pose_keys, input_key, m_offset,
                                                         gtsam::noiseModel::Gaussian::Covariance(gtsam::Matrix::Constant(1, 1, m_length_sigma * m_length_sigma)))};
        }

    private:
        /// diskArclengths(termination, discretization). @throws std::runtime_error for a null prior.
        static std::vector<double> evenDisks(const std::shared_ptr<PriorT> &prior, double termination, double discretization)
        {
            if (!prior)
                throw std::runtime_error("KinematicTendon requires a non-null prior.");
            const double length = prior->state()->arclengths.back();
            if (termination > length + TOLERANCE)
                throw std::invalid_argument("KinematicTendon: the termination (" + std::to_string(termination) +
                                            ") is past the robot's length (" + std::to_string(length) + ").");
            return diskArclengths(termination, discretization);
        }

        /// `disks`, validated (see the constructors).
        static std::vector<double> checkedDisks(const std::shared_ptr<PriorT> &prior, double radius, double angle, std::vector<double> disks)
        {
            if (!prior)
                throw std::runtime_error("KinematicTendon requires a non-null prior.");
            if (!(radius >= 0.0) || !std::isfinite(radius))
                throw std::invalid_argument("KinematicTendon: the radius must be non-negative.");
            if (!std::isfinite(angle))
                throw std::invalid_argument("KinematicTendon: the angle must be finite.");
            if (disks.size() < 2)
                throw std::invalid_argument("KinematicTendon: a tendon needs at least two routing disks.");
            const double length = prior->state()->arclengths.back();
            for (size_t i = 0; i < disks.size(); ++i)
            {
                if (!std::isfinite(disks[i]) || disks[i] < -TOLERANCE || disks[i] > length + TOLERANCE)
                    throw std::invalid_argument("KinematicTendon: the disk arclength " + std::to_string(disks[i]) +
                                                " is outside the robot [0, " + std::to_string(length) + "].");
                if (i > 0 && !(disks[i] > disks[i - 1] + TOLERANCE))
                    throw std::invalid_argument("KinematicTendon: the disk arclengths must be strictly ascending.");
            }
            return disks;
        }

        std::shared_ptr<PriorT> mp_prior;
        double m_radius;
        double m_angle;
        double m_termination;
        std::optional<double> m_discretization;
        gtsam::Point3 m_offset;
        double m_length_sigma = DEFAULT_LENGTH_SIGMA;
    };
} // namespace Spacetime::Actuators

#endif // KINEMATICTENDON_HPP
