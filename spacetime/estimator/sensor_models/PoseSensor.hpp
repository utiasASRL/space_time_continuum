#ifndef PoseSensor_HPP
#define PoseSensor_HPP

#include <Eigen/Dense>
#include <concepts>
#include <memory>
#include <utility>

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>

#include <spacetime/Types.hpp>

namespace Spacetime::Sensors
{
    using MeasurementType = Types::MeasurementBase;
    namespace
    {
        /// Node values with a pose() and an interpolated Pose() key.
        template <typename ValuesT>
        concept PoseSensorValuesType = Types::ValuesType<ValuesT> && requires(ValuesT n) {
            typename ValuesT::PoseType;
            typename ValuesT::InterpolatedKeys;
            { n.pose() } -> std::convertible_to<typename ValuesT::PoseType>;
            { ValuesT::InterpolatedKeys::Pose() } -> std::convertible_to<gtsam::Key>;
        };
    };

    /**
     * @brief Pose sensor (e.g. vicon or aurora; the paper's eqs. (27)-(29), there for T_bi): measures the
     *        pose T_ib of the robot at a given arclength and time. Measurement values are
     *        pose matrices (e.g. 4x4), covariances dim x dim in the pose's tangent order
     *        (GTSAM [rotational, translational] for Pose3).
     */
    template <PoseSensorValuesType ValuesT>
    class PoseSensor : public Types::SensorBase
    {
        using Pose = ValuesT::PoseType;
        static constexpr int dim = gtsam::traits<Pose>::dimension;
        using TangentVector = typename gtsam::traits<Pose>::TangentVector;
        using ErrorVector = Eigen::Matrix<double, dim, 1>;

        // Check that Pose type is a testable Lie group
        GTSAM_CONCEPT_ASSERT(gtsam::IsTestable<Pose>);
        // We currently support vector spaces and Lie groups
        static_assert(std::is_same_v<typename gtsam::traits<Pose>::structure_category,
                                     gtsam::lie_group_tag> ||
                          std::is_same_v<typename gtsam::traits<Pose>::structure_category,
                                         gtsam::vector_space_tag>,
                      "Pose type must be either a Lie group or vector space");

        GTSAM_CONCEPT_ASSERT(gtsam::IsLieGroup<Pose>);

    public:
        /**
         * @brief Measurement factor built by PoseSensor::getFactor: the pose-logmap
         *        error e = Log(T^-1 T_meas) against a fixed measured pose, referencing only
         *        ValuesT::InterpolatedKeys::Pose() -- whatever key the node type
         *        this sensor is instantiated for uses to stand in for its
         *        interpolated pose component (this sensor's error does not depend
         *        on any of the interpolated node's other components) -- evaluated
         *        as the inner factor of a Priors::InterpolationFactor.
         */
        class Factor : public gtsam::NoiseModelFactorN<Pose>
        {
        public:
            using Base = gtsam::NoiseModelFactorN<Pose>;

            /// `mask` is dim x 1; a 0 entry ignores that component.
            Factor(const gtsam::SharedNoiseModel &noise_model, const Pose &measured, const Eigen::MatrixXi &mask)
                : Base(noise_model, ValuesT::InterpolatedKeys::Pose()), m_measured(measured), m_mask(mask) {}

            /// The masked error Log(pose^-1 T_meas), with H its Jacobian in pose.
            gtsam::Vector evaluateError(const Pose &pose, gtsam::OptionalMatrixType H = nullptr) const override
            {
                Eigen::Matrix<double, dim, dim> dT_em_de, dxi_em_dT_em;
                Pose T_em = pose.between(m_measured, &dT_em_de, nullptr);
                ErrorVector e = gtsam::traits<Pose>::Logmap(T_em, &dxi_em_dT_em);

                for (unsigned int i = 0; i < dim; i++)
                    if (m_mask(i, 0) == 0)
                        e(i, 0) = 0;

                if (H)
                {
                    Eigen::Matrix<double, dim, dim> S = dxi_em_dT_em * dT_em_de;
                    for (unsigned int i = 0; i < dim; i++)
                        if (m_mask(i, 0) == 0)
                            S.row(i).setZero();
                    *H = S;
                }
                return e;
            }

        private:
            Pose m_measured;
            Eigen::MatrixXi m_mask;
        };

        /**
         * @brief A sensor measuring the state of `stateful`.
         * @param weight Unused: each measurement carries its own covariance.
         */
        PoseSensor(Eigen::Matrix<double, 6, 6> weight, Types::SensorBase::pStatefulType stateful)
            : Types::SensorBase(std::move(stateful)) {}

        ~PoseSensor() = default;

        /// The Factor for `measurement`, with a Gaussian noise model of its covariance.
        gtsam::NoiseModelFactor::shared_ptr getFactor(const pMeasurementType measurement) const override
        {
            auto noise_model = gtsam::noiseModel::Gaussian::Covariance(measurement->covariance);
            return std::make_shared<Factor>(noise_model, Pose(measurement->value), measurement->mask);
        }
    };
} // namespace Spacetime::Sensors
#endif // PoseSensor_HPP