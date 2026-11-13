/**
 * @file GroundTruthExport.hpp
 * @brief Exports the estimate queried at every ground-truth sample, in the file format
 *        of the legacy estimator's exportErrorData (logs/error_pred.txt and
 *        logs/error_pred_cov.txt), so both versions can be scored by the same metrics
 *        script (tests/regression/common/regression_metrics.py).
 * @author Spencer Teetaert
 */

#ifndef GROUND_TRUTH_EXPORT_HPP
#define GROUND_TRUTH_EXPORT_HPP

#include <algorithm>
#include <filesystem>
#include <limits>
#include <fstream>
#include <functional>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/PosteriorCovariance.hpp>
#include <spacetime/estimator/types/SensorBase.hpp>

namespace Spacetime::Utilities
{
    /**
     * @brief Swap the two halves of every 6-vector block, converting between the GTSAM
     *        [rotational, translational] and the legacy [translational, rotational]
     *        tangent orderings (the swap is its own inverse).
     */
    template <int Blocks>
    Eigen::Matrix<double, 6 * Blocks, 6 * Blocks> tangentOrderSwap()
    {
        Eigen::Matrix<double, 6 * Blocks, 6 * Blocks> P = Eigen::Matrix<double, 6 * Blocks, 6 * Blocks>::Zero();
        for (int b = 0; b < Blocks; b++)
        {
            P.template block<3, 3>(6 * b, 6 * b + 3).setIdentity();
            P.template block<3, 3>(6 * b + 3, 6 * b).setIdentity();
        }
        return P;
    }

    /**
     * @brief Query the estimate at every ground-truth sample and write it out in the
     *        legacy format:
     *         - error_pred.txt: "time,arclength,x1..x6,e1..e6,v1..v6," with x = Log(T_ib),
     *           and eps, varpi, all in the legacy [translational, rotational] order.
     *         - error_pred_cov.txt: "time,arclength," then the row-major 18 x 18 posterior
     *           covariance of [pose, eps, varpi] in the same order, for the samples where
     *           it is available.
     *        With T_ib and a right perturbation, the pose error Log(T_est^-1 T_gt) and its
     *        covariance equal the legacy T_bi left-perturbation ones up to this reordering.
     *        Samples outside the estimate's extent (see StatefulModel::query) are skipped.
     *
     *        For a prior whose node holds no velocity (e.g. WNOAQuasiStatic, pose and
     *        strain) or neither strain nor velocity (e.g. WNOV, pose only), the format is
     *        kept: the missing parts are written as nan, and the covariance holds the
     *        leading 12 x 12 (or 6 x 6) block with nan elsewhere.
     * @tparam PriorT The prior (e.g. WNOA::Prior<Pose>, WNOV::Prior<Pose>), whose ValuesType
     *         provides pose() and, optionally, epsilon() and, with it, varpi().
     * @param prior The prior whose interpolation queries the estimate.
     * @param values The estimate (e.g. WindowHistory::values or Estimator::getCurrentValues).
     * @param posterior_at The posterior covariance to query a sample at time t with
     *        (e.g. WindowHistory::posteriorAt), or nullptr for none.
     * @param ground_truth_sensors Sensors whose queued measurements give the sample
     *        (time, arclength) pairs; their queues are copied, not drained.
     * @param logs_dir Output directory (created if missing).
     * @return The number of samples written to error_pred.txt.
     * @throws std::runtime_error if the output files cannot be opened.
     */
    template <typename PriorT>
    std::size_t exportGroundTruthPredictions(
        const PriorT &prior,
        const gtsam::Values &values,
        const std::function<const PosteriorCovariance *(double)> &posterior_at,
        const std::vector<std::shared_ptr<Types::SensorBase>> &ground_truth_sensors,
        const std::string &logs_dir)
    {
        using ValuesType = typename PriorT::ValuesType;
        using Pose = typename ValuesType::PoseType;
        static_assert(gtsam::traits<Pose>::dimension == 6, "The legacy export format is defined for SE(3) only.");
        constexpr bool has_strain = requires(const ValuesType &v) { v.epsilon(); };
        constexpr bool has_velocity = requires(const ValuesType &v) { v.varpi(); };
        constexpr int node_dim = 6 * (1 + int(has_strain) + int(has_velocity));
        const double nan = std::numeric_limits<double>::quiet_NaN();

        // Ground-truth sensors are never drained by an estimator, so their queues still
        // hold every sample; copy them rather than consuming them.
        std::vector<Types::SensorBase::pMeasurementType> samples;
        for (const auto &sensor : ground_truth_sensors)
        {
            const auto queued = sensor->peekAll();
            samples.insert(samples.end(), queued.begin(), queued.end());
        }
        std::stable_sort(samples.begin(), samples.end(), [](const auto &a, const auto &b)
                         { return a->time < b->time || (a->time == b->time && a->arclength < b->arclength); });

        std::filesystem::create_directories(logs_dir);
        std::ofstream mean_file(std::filesystem::path(logs_dir) / "error_pred.txt");
        std::ofstream cov_file(std::filesystem::path(logs_dir) / "error_pred_cov.txt");
        if (!mean_file || !cov_file)
            throw std::runtime_error("exportGroundTruthPredictions: cannot open output files in " + logs_dir);
        mean_file.precision(17);
        cov_file.precision(17);
        mean_file << "time,arclength,x1,x2,x3,x4,x5,x6,e1,e2,e3,e4,e5,e6,v1,v2,v3,v4,v5,v6,\n";
        cov_file << "time,arclength,\n";

        const Eigen::Matrix<double, 6, 6> swap6 = tangentOrderSwap<1>();
        const Eigen::Matrix<double, node_dim, node_dim> swap_node = tangentOrderSwap<node_dim / 6>();

        std::size_t written = 0;
        for (const auto &sample : samples)
        {
            const PosteriorCovariance *posterior = posterior_at ? posterior_at(sample->time) : nullptr;
            Eigen::MatrixXd covariance;
            auto queried = std::dynamic_pointer_cast<ValuesType>(
                prior.query(sample->time, sample->arclength, values, nullptr, posterior, posterior ? &covariance : nullptr));
            if (!queried)
                continue;

            Eigen::Matrix<double, 18, 1> row = Eigen::Matrix<double, 18, 1>::Constant(nan);
            row.template head<6>() = swap6 * gtsam::traits<Pose>::Logmap(queried->pose());
            if constexpr (has_strain)
                row.template segment<6>(6) = swap6 * queried->epsilon();
            if constexpr (has_velocity)
                row.template tail<6>() = swap6 * queried->varpi();

            mean_file << sample->time << "," << sample->arclength << ",";
            for (int i = 0; i < 18; i++)
                mean_file << row(i) << ",";
            mean_file << "\n";
            written++;

            if (covariance.rows() == node_dim && covariance.cols() == node_dim)
            {
                Eigen::Matrix<double, 18, 18> legacy_covariance = Eigen::Matrix<double, 18, 18>::Constant(nan);
                legacy_covariance.template topLeftCorner<node_dim, node_dim>() = swap_node * covariance * swap_node.transpose();
                cov_file << sample->time << "," << sample->arclength << ",";
                for (int r = 0; r < 18; r++)
                    for (int c = 0; c < 18; c++)
                        cov_file << legacy_covariance(r, c) << ",";
                cov_file << "\n";
            }
        }
        return written;
    }

    /// As above, with a single posterior (or nullptr) for every sample.
    template <typename PriorT>
    std::size_t exportGroundTruthPredictions(
        const PriorT &prior,
        const gtsam::Values &values,
        const PosteriorCovariance *posterior,
        const std::vector<std::shared_ptr<Types::SensorBase>> &ground_truth_sensors,
        const std::string &logs_dir)
    {
        return exportGroundTruthPredictions(
            prior, values, [posterior](double) { return posterior; }, ground_truth_sensors, logs_dir);
    }
}

#endif // GROUND_TRUTH_EXPORT_HPP
