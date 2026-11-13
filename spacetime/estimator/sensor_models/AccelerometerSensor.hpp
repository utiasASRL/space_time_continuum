#ifndef ACCELEROMETERSENSOR_HPP
#define ACCELEROMETERSENSOR_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/base/Lie.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/bias_models/Bias.hpp>
#include <spacetime/estimator/prior_models/InterpolatedNode.hpp>
#include <spacetime/estimator/prior_models/TwoSidedInterpolationFactor.hpp>
#include <spacetime/estimator/utilities/functions.hpp>
#include <spacetime/estimator/utilities/logger.hpp>

namespace Spacetime::Sensors
{
    /// Parameters of an AccelerometerSensor (and of its AccelerometerFactor).
    struct AccelerometerOptions
    {
        /// Pose T_bs of the accelerometer in the robot's body frame at its arclength.
        gtsam::Pose3 offset = gtsam::Pose3();
        /// Rotate each sample by SLERP between the interval's two node rotations, instead of
        /// holding the first node's rotation over the interval (see AccelerometerFactor).
        bool use_slerp = false;
        /// Gravity in the inertial frame [m/s^2]; the accelerometer reads specific force.
        gtsam::Vector3 gravity = gtsam::Vector3(0.0, 0.0, -9.81);
        /// Estimate an accelerometer bias (a Biases::BiasModel owned by the sensor).
        bool estimate_bias = true;
        /// The bias's prior and random walk, used when estimate_bias is set.
        Biases::BiasOptions<gtsam::Vector3> bias = defaultBias();

        /// Zero initial bias, with a 0.1 m/s^2 prior std and a 1e-3 m/s^2/sqrt(s) random walk.
        static Biases::BiasOptions<gtsam::Vector3> defaultBias()
        {
            Biases::BiasOptions<gtsam::Vector3> bias;
            bias.initial = gtsam::Vector3::Zero();
            bias.initial_covariance = 1e-2 * gtsam::Matrix3::Identity();
            bias.random_walk_psd = 1e-6 * gtsam::Matrix3::Identity();
            return bias;
        }
    };

    /**
     * @class AccelerometerPreintegration
     * @brief The accelerometer samples of one interval [t_start, t_end] between two node
     *        rows, integrated with a zero-order hold: each sample holds from its time until
     *        the next sample's (or t_end), clipped to the interval, so the last sample before
     *        t_start (carried over from the previous interval) covers its start.
     *
     * Each held sample keeps its duration dt_j, the fraction tau_j of the interval at its
     * midpoint, its reading and (given a bias-weights function) the linear interpolation
     * weights of the bias nodes at that midpoint. The interval-wide sums that make the
     * factor's hold mode O(1) to evaluate are accumulated as samples are added. Readings and
     * covariances stay in the sensor frame.
     */
    class AccelerometerPreintegration
    {
    public:
        /// Weights of each bias node key in a linearly interpolated bias (see Biases::BiasModel::weights).
        using BiasWeights = std::vector<std::pair<gtsam::Key, double>>;
        /// The bias weights at a time.
        using BiasWeightsFn = std::function<BiasWeights(double time)>;

        /// One held sample.
        struct Sample
        {
            double tau;              ///< Fraction of the interval at the held span's midpoint.
            double dt;               ///< Held duration [s].
            gtsam::Vector3 measured; ///< Reading, sensor frame [m/s^2].
            BiasWeights bias_weights;
        };

        /**
         * @param bias_weights Optional bias weights; without them the readings are taken as unbiased.
         * @throws std::invalid_argument unless t_end > t_start.
         */
        AccelerometerPreintegration(double t_start, double t_end, BiasWeightsFn bias_weights = nullptr)
            : m_t_start(t_start), m_t_end(t_end), m_bias_weights_fn(std::move(bias_weights))
        {
            if (!(t_end > t_start + TOLERANCE))
                throw std::invalid_argument("AccelerometerPreintegration: the interval must end after it starts.");
        }

        /**
         * @brief Adds a sample read at `time`, held until the next one (or t_end). Samples
         *        must come in time order; one before t_start only covers the interval's start.
         * @throws std::invalid_argument for a sample older than the previous one, at or after
         *         t_end, or added after close().
         */
        void addSample(double time, const gtsam::Vector3 &measured, const gtsam::Matrix3 &covariance)
        {
            if (m_closed)
                throw std::invalid_argument("AccelerometerPreintegration: cannot add a sample after close().");
            if (time >= m_t_end - TOLERANCE)
                throw std::invalid_argument("AccelerometerPreintegration: a sample at or after the interval's end belongs to the next interval.");
            if (m_pending && time < m_pending->time - TOLERANCE)
                throw std::invalid_argument("AccelerometerPreintegration: samples must be added in time order.");
            if (m_pending)
                hold(*m_pending, time);
            m_pending = Pending{time, measured, covariance};
        }

        /// Holds the last sample until t_end. Idempotent; no sample can be added after it.
        void close()
        {
            if (m_closed)
                return;
            if (m_pending)
                hold(*m_pending, m_t_end);
            m_pending.reset();
            m_closed = true;
        }

        /// Closed, and held samples cover the whole interval.
        bool complete() const
        {
            return m_closed && m_covered_from && *m_covered_from <= m_t_start + TOLERANCE && std::abs(m_covered - deltaT()) < TOLERANCE;
        }

        double startTime() const { return m_t_start; }
        double endTime() const { return m_t_end; }
        double deltaT() const { return m_t_end - m_t_start; }
        /// Total held time.
        double coveredTime() const { return m_covered; }

        const std::vector<Sample> &samples() const { return m_samples; }
        /// Sum of dt_j * reading_j.
        const gtsam::Vector3 &sumMeasured() const { return m_sum_measured; }
        /// Sums of dt_j (1 - tau_j)^2, dt_j tau_j (1 - tau_j) and dt_j tau_j^2: the weights of
        /// the centripetal lever-arm term with the angular velocity linear in tau.
        double w0() const { return m_w0; }
        double w01() const { return m_w01; }
        double w1() const { return m_w1; }
        /// Sum of dt_j * weight, per bias key, in order of first appearance.
        const BiasWeights &biasWeights() const { return m_bias_weights; }
        gtsam::KeyVector biasKeys() const
        {
            gtsam::KeyVector keys;
            for (const auto &[key, weight] : m_bias_weights)
                keys.push_back(key);
            return keys;
        }
        /// Covariance of sumMeasured(): the sum of dt_j^2 Sigma_j (sensor frame).
        const gtsam::Matrix3 &covariance() const { return m_covariance; }

    private:
        struct Pending
        {
            double time;
            gtsam::Vector3 measured;
            gtsam::Matrix3 covariance;
        };

        void hold(const Pending &sample, double until)
        {
            const double from = std::max(sample.time, m_t_start), to = std::min(until, m_t_end);
            if (to <= from)
                return;
            if (!m_covered_from)
                m_covered_from = from;

            const double dt = to - from, mid = 0.5 * (from + to), tau = (mid - m_t_start) / deltaT();
            Sample held{tau, dt, sample.measured, {}};
            if (m_bias_weights_fn)
                held.bias_weights = m_bias_weights_fn(mid);
            for (const auto &[key, weight] : held.bias_weights)
            {
                auto it = std::find_if(m_bias_weights.begin(), m_bias_weights.end(), [key](const auto &entry) { return entry.first == key; });
                if (it == m_bias_weights.end())
                    m_bias_weights.emplace_back(key, dt * weight);
                else
                    it->second += dt * weight;
            }

            m_sum_measured += dt * sample.measured;
            m_w0 += dt * (1.0 - tau) * (1.0 - tau);
            m_w01 += dt * tau * (1.0 - tau);
            m_w1 += dt * tau * tau;
            m_covariance += dt * dt * sample.covariance;
            m_covered += dt;
            m_samples.push_back(std::move(held));
        }

        double m_t_start, m_t_end;
        BiasWeightsFn m_bias_weights_fn;
        std::optional<Pending> m_pending;
        std::optional<double> m_covered_from;
        bool m_closed = false;

        std::vector<Sample> m_samples;
        gtsam::Vector3 m_sum_measured = gtsam::Vector3::Zero();
        double m_w0 = 0.0, m_w01 = 0.0, m_w1 = 0.0, m_covered = 0.0;
        BiasWeights m_bias_weights;
        gtsam::Matrix3 m_covariance = gtsam::Matrix3::Zero();
    };

    /**
     * @class AccelerometerFactor
     * @brief The preintegrated accelerometer factor between two nodes k and k+1 (poses T_ib,
     *        body twists varpi = [omega, nu], GTSAM [rotational, translational] order): the
     *        body-frame velocity increment the readings imply against the one the nodes give.
     *
     * With T_bs = (R_bs, r) the accelerometer's offset, a reading a_j (specific force,
     * sensor frame, bias b_j) gives the body-frame specific force at the body origin
     *
     *     f_j = R_bs (a_j - b_j) - (alpha x r + w_j x (w_j x r)),
     *
     * with alpha = (omega_{k+1} - omega_k) / dt and w_j = (1 - tau_j) omega_k + tau_j omega_{k+1}.
     * The 3x1 residual is
     *
     *     e = R_k^T R_{k+1} nu_{k+1} - nu_k - R_k^T g dt - sum_j dR_j f_j dt_j,
     *
     * where dR_j = I holds R_k over the interval (the default: the sums are preintegrated, so
     * the factor is O(1) to evaluate), or dR_j = Exp(tau_j Log(R_k^T R_{k+1})) with
     * AccelerometerOptions::use_slerp (O(samples)). b_j is the bias interpolated linearly between the
     * bias nodes (passed-through keys after the four node keys); without bias keys the
     * readings are taken as unbiased.
     *
     * The noise is R_bs (sum_j dt_j^2 Sigma_j) R_bs^T, fixed at construction (in SLERP mode
     * this neglects dR_j, exact for an isotropic Sigma_j). Jacobians are analytic.
     */
    class AccelerometerFactor : public gtsam::NoiseModelFactor
    {
    public:
        using Base = gtsam::NoiseModelFactor;
        using This = AccelerometerFactor;
        using Vector6 = Eigen::Matrix<double, 6, 1>;

        using Base::error;
        using Base::unwhitenedError;

        /**
         * @param pose_k,velocity_k Node k's pose (gtsam::Pose3) and twist (Vector6) keys.
         * @param pose_k1,velocity_k1 Node k+1's.
         * @param preintegration The interval's samples; its bias keys are appended to this factor's keys.
         * @throws std::invalid_argument if `preintegration` is not complete().
         */
        AccelerometerFactor(gtsam::Key pose_k, gtsam::Key velocity_k, gtsam::Key pose_k1, gtsam::Key velocity_k1,
                  const AccelerometerPreintegration &preintegration, const AccelerometerOptions &options)
            : Base(noiseModelOf(preintegration, options)), m_preintegration(preintegration),
              m_R_bs(options.offset.rotation().matrix()), m_r(options.offset.translation()), m_gravity(options.gravity),
              m_use_slerp(options.use_slerp)
        {
            this->keys_ = {pose_k, velocity_k, pose_k1, velocity_k1};
            for (const gtsam::Key key : preintegration.biasKeys())
                this->keys_.push_back(key);
            m_rotated_sum = m_R_bs * preintegration.sumMeasured();
        }

        void print(const std::string &s = "", const gtsam::KeyFormatter &keyFormatter = gtsam::DefaultKeyFormatter) const override
        {
            std::cout << s << "AccelerometerFactor over [" << m_preintegration.startTime() << ", " << m_preintegration.endTime() << "] ("
                      << m_preintegration.samples().size() << " samples" << (m_use_slerp ? ", SLERP" : "") << ") on ";
            for (const auto &k : this->keys())
                std::cout << keyFormatter(k) << " ";
            std::cout << std::endl;
        }

        /// Same type, keys and noise model, and the same preintegrated readings.
        bool equals(const gtsam::NonlinearFactor &expected, double tol = 1e-9) const override
        {
            const This *e = dynamic_cast<const This *>(&expected);
            return e != nullptr && Base::equals(*e, tol) && m_use_slerp == e->m_use_slerp &&
                   m_rotated_sum.isApprox(e->m_rotated_sum, tol) && std::abs(m_preintegration.deltaT() - e->m_preintegration.deltaT()) < tol;
        }

        const AccelerometerPreintegration &preintegration() const { return m_preintegration; }

        /// The residual e (see the class description), with Jacobians w.r.t. every key.
        gtsam::Vector unwhitenedError(const gtsam::Values &values, gtsam::OptionalMatrixVecType H = nullptr) const override
        {
            using gtsam::skewSymmetric;
            const gtsam::Rot3 R0 = values.at<gtsam::Pose3>(keys_[0]).rotation(), R1 = values.at<gtsam::Pose3>(keys_[2]).rotation();
            const Vector6 &varpi0 = values.at<Vector6>(keys_[1]), &varpi1 = values.at<Vector6>(keys_[3]);
            const gtsam::Vector3 w0 = varpi0.head<3>(), w1 = varpi1.head<3>(), nu0 = varpi0.tail<3>(), nu1 = varpi1.tail<3>();
            const double dt = m_preintegration.deltaT();
            const auto &bias_weights = m_preintegration.biasWeights();
            const size_t num_bias = bias_weights.size();
            const gtsam::Matrix3 r_hat = skewSymmetric(m_r);
            // d/da of a x (a x r), and of a symmetric pair a x (b x r) + b x (a x r) w.r.t. a (at b)
            const auto D = [&](const gtsam::Vector3 &a) -> gtsam::Matrix3 { return -skewSymmetric(a.cross(m_r)) - skewSymmetric(a) * r_hat; };

            // The nodes' velocity increment: R_k^T R_{k+1} nu_{k+1} - nu_k - R_k^T g dt.
            gtsam::Matrix3 H1_R1, H1_nu1, H0_unrotate, H0_p, Hg_R0;
            const gtsam::Vector3 v1 = R1.rotate(nu1, H1_R1, H1_nu1);
            gtsam::Vector3 e = R0.unrotate(v1, H0_unrotate, H0_p) - nu0 - dt * R0.unrotate(m_gravity, Hg_R0);
            gtsam::Matrix3 dR0 = H0_unrotate - dt * Hg_R0, dR1 = H0_p * H1_R1, dnu1 = H0_p * H1_nu1;
            gtsam::Matrix3 dw0 = gtsam::Matrix3::Zero(), dw1 = gtsam::Matrix3::Zero();
            std::vector<gtsam::Matrix3> db(num_bias, gtsam::Matrix3::Zero());

            if (!m_use_slerp)
            {
                // Minus the preintegrated specific force: R_bs (sum a dt - sum W_i b_i) - lever arm.
                e -= m_rotated_sum;
                for (size_t i = 0; i < num_bias; i++)
                {
                    db[i] = bias_weights[i].second * m_R_bs;
                    e += db[i] * values.at<gtsam::Vector3>(bias_weights[i].first);
                }
                const double w0_sum = m_preintegration.w0(), w01_sum = m_preintegration.w01(), w1_sum = m_preintegration.w1();
                const double alpha_scale = m_preintegration.coveredTime() / dt;
                const auto A = [&](const gtsam::Vector3 &a, const gtsam::Vector3 &b) -> gtsam::Vector3 { return a.cross(b.cross(m_r)); };
                e += alpha_scale * (w1 - w0).cross(m_r) + w0_sum * A(w0, w0) + w01_sum * (A(w0, w1) + A(w1, w0)) + w1_sum * A(w1, w1);
                dw0 = alpha_scale * r_hat + w0_sum * D(w0) + w01_sum * D(w1);
                dw1 = -alpha_scale * r_hat + w01_sum * D(w0) + w1_sum * D(w1);
            }
            else
            {
                const gtsam::Vector3 alpha = (w1 - w0) / dt;
                for (const auto &sample : m_preintegration.samples())
                {
                    gtsam::Vector3 b = gtsam::Vector3::Zero();
                    for (const auto &[key, weight] : sample.bias_weights)
                        b += weight * values.at<gtsam::Vector3>(key);
                    const gtsam::Vector3 wj = (1.0 - sample.tau) * w0 + sample.tau * w1;
                    const gtsam::Vector3 u = m_R_bs * (sample.measured - b) - (alpha.cross(m_r) + wj.cross(wj.cross(m_r)));

                    gtsam::Matrix3 Hj_R0, Hj_R1, Hu_Rj, Hu_u, Hq_R0, Hq_p;
                    const gtsam::Rot3 Rj = gtsam::interpolate<gtsam::Rot3>(R0, R1, sample.tau, Hj_R0, Hj_R1);
                    const gtsam::Vector3 q = R0.unrotate(Rj.rotate(u, Hu_Rj, Hu_u), Hq_R0, Hq_p);

                    e -= sample.dt * q;
                    dR0 -= sample.dt * (Hq_R0 + Hq_p * Hu_Rj * Hj_R0);
                    dR1 -= sample.dt * Hq_p * Hu_Rj * Hj_R1;
                    // dq/du = R_k^T R_j; u depends on omega through the lever arm and on the bias.
                    const gtsam::Matrix3 dq_du = Hq_p * Hu_u;
                    const gtsam::Matrix3 dl_dw0 = r_hat / dt + (1.0 - sample.tau) * D(wj), dl_dw1 = -r_hat / dt + sample.tau * D(wj);
                    dw0 += sample.dt * dq_du * dl_dw0;
                    dw1 += sample.dt * dq_du * dl_dw1;
                    for (const auto &[key, weight] : sample.bias_weights)
                    {
                        const size_t i = biasIndex(key);
                        db[i] += sample.dt * weight * dq_du * m_R_bs;
                    }
                }
            }

            if (H)
            {
                Eigen::Matrix<double, 3, 6> H_pose0 = Eigen::Matrix<double, 3, 6>::Zero(), H_pose1 = Eigen::Matrix<double, 3, 6>::Zero();
                Eigen::Matrix<double, 3, 6> H_varpi0, H_varpi1;
                H_pose0.leftCols<3>() = dR0;
                H_pose1.leftCols<3>() = dR1;
                H_varpi0 << dw0, -gtsam::Matrix3::Identity();
                H_varpi1 << dw1, dnu1;
                (*H)[0] = H_pose0;
                (*H)[1] = H_varpi0;
                (*H)[2] = H_pose1;
                (*H)[3] = H_varpi1;
                for (size_t i = 0; i < num_bias; i++)
                    (*H)[4 + i] = db[i];
            }
            return e;
        }

    private:
        static gtsam::SharedNoiseModel noiseModelOf(const AccelerometerPreintegration &preintegration, const AccelerometerOptions &options)
        {
            if (!preintegration.complete())
                throw std::invalid_argument("AccelerometerFactor: the preintegrated samples do not cover the whole interval.");
            const gtsam::Matrix3 R_bs = options.offset.rotation().matrix();
            return gtsam::noiseModel::Gaussian::Covariance(R_bs * preintegration.covariance() * R_bs.transpose());
        }

        size_t biasIndex(gtsam::Key key) const
        {
            const auto &weights = m_preintegration.biasWeights();
            for (size_t i = 0; i < weights.size(); i++)
                if (weights[i].first == key)
                    return i;
            throw std::logic_error("AccelerometerFactor: a sample's bias key is not one of the factor's.");
        }

        AccelerometerPreintegration m_preintegration;
        gtsam::Matrix3 m_R_bs;
        gtsam::Vector3 m_r, m_gravity;
        bool m_use_slerp;
        gtsam::Vector3 m_rotated_sum; ///< R_bs * sum_j dt_j a_j
    };

    /// Priors an AccelerometerSensor can measure: grid priors whose node has a gtsam::Pose3 pose and a velocity.
    template <typename PriorT>
    concept AccelerometerPriorType = requires(const PriorT &prior, double time, double arclength) {
        typename PriorT::InterpolatorType;
        typename PriorT::ValuesType;
        { PriorT::InterpolatorType::InterpolatedKeys::Pose() } -> std::convertible_to<gtsam::Key>;
        { PriorT::InterpolatorType::InterpolatedKeys::Velocity() } -> std::convertible_to<gtsam::Key>;
        { prior.interpolatedNode(time, arclength) };
        { prior.windowStart() } -> std::convertible_to<double>;
        prior.state()->arclengths;
        prior.state()->times();
    } && std::is_same_v<std::decay_t<decltype(std::declval<typename PriorT::ValuesType>().pose())>, gtsam::Pose3>;

    /**
     * @class AccelerometerSensor
     * @brief A 3-axis accelerometer at a fixed arclength of a robot (or end effector),
     *        reading specific force [m/s^2] in its own frame (offset T_bs from the body frame).
     *        Its samples (3x1 values, 3x3 covariances) are preintegrated between consecutive
     *        node rows into one AccelerometerFactor per interval, evaluated on the two rows'
     *        interpolated nodes at the sensor's arclength through a
     *        Priors::TwoSidedInterpolationFactor.
     *
     * With AccelerometerOptions::estimate_bias, the sensor owns a Biases::BiasModel<gtsam::Vector3>
     * (one bias node per estimation time, a random walk) and delegates its extendState,
     * onMarginalization, lockedValues and covarianceKeyGroups to it, so adding the sensor to
     * the Estimator adds its bias states.
     *
     * An interval's factor is built once its end row exists, from the samples drained up to
     * then. Dropped, with a warning: samples at another arclength, with a masked component,
     * or arriving after their interval was built; and intervals the samples do not cover
     * (e.g. the first one after the accelerometer starts) or whose first row was already marginalized.
     *
     * @tparam PriorT The prior extending the measured state (e.g. WNOA::Prior<gtsam::Pose3>,
     *         Joints::EndEffectorPrior<gtsam::Pose3>).
     */
    template <AccelerometerPriorType PriorT>
    class AccelerometerSensor : public Types::SensorBase
    {
    public:
        using InterpolatorType = typename PriorT::InterpolatorType;
        using Keys = typename InterpolatorType::InterpolatedKeys;
        using NodeType = Priors::InterpolatedNode<InterpolatorType>;
        using FactorType = Priors::TwoSidedInterpolationFactor<InterpolatorType, InterpolatorType>;
        using BiasModelType = Biases::BiasModel<gtsam::Vector3>;

        /**
         * @param prior The prior extending the measured state.
         * @param arclength The sensor's arclength, within the prior's nodes.
         * @throws std::invalid_argument for a null prior or an arclength outside its nodes.
         */
        AccelerometerSensor(std::shared_ptr<PriorT> prior, double arclength, const AccelerometerOptions &options = AccelerometerOptions())
            : Types::SensorBase(checked(prior)), m_prior(std::move(prior)), m_arclength(arclength), m_options(options)
        {
            const auto &arclengths = m_prior->state()->arclengths;
            if (arclengths.empty() || arclength < arclengths.front() - TOLERANCE || arclength > arclengths.back() + TOLERANCE)
                throw std::invalid_argument("AccelerometerSensor: arclength " + std::to_string(arclength) + " lies outside the robot's nodes.");
            if (m_options.estimate_bias)
                m_bias = std::make_shared<BiasModelType>(m_options.bias, arclength);
        }

        double arclength() const { return m_arclength; }
        const AccelerometerOptions &options() const { return m_options; }
        /// The estimated accelerometer bias (e.g. to query it), or nullptr without AccelerometerOptions::estimate_bias.
        const std::shared_ptr<BiasModelType> &biasModel() const { return m_bias; }

        /// No per-measurement factor: samples are preintegrated per interval (see getNewFactors).
        gtsam::NoiseModelFactor::shared_ptr getFactor(const pMeasurementType measurement) const override
        {
            (void)measurement;
            return nullptr;
        }

        /// Adds the bias node at `time`, if the bias is estimated.
        Factors extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override
        {
            return m_bias ? m_bias->extendState(km, time, current_values, new_values) : Factors();
        }

        void onMarginalization(double cutoff, const gtsam::Values &values) override
        {
            if (m_bias)
                m_bias->onMarginalization(cutoff, values);
        }

        const gtsam::Values &lockedValues() const override { return m_bias ? m_bias->lockedValues() : Types::SensorBase::lockedValues(); }

        std::vector<gtsam::KeyVector> covarianceKeyGroups(const gtsam::Values &values) const override
        {
            return m_bias ? m_bias->covarianceKeyGroups(values) : std::vector<gtsam::KeyVector>();
        }

        /// Buffers the samples drained up to `time`, and returns one factor per interval between rows that now ends at or before it.
        Factors getNewFactors(double time, gtsam::Values &new_values) override
        {
            (void)new_values;
            for (const pMeasurementType &measurement : drainUpTo(time))
                buffer(measurement);

            Factors factors;
            const std::vector<double> &rows = m_prior->state()->times();
            for (; m_next_row < rows.size() && rows[m_next_row] <= time + TOLERANCE; m_next_row++)
            {
                if (m_next_row == 0)
                    continue; // the first row ends no interval
                if (auto factor = intervalFactor(rows[m_next_row - 1], rows[m_next_row]))
                    factors.push_back(factor);
                m_closed_until = rows[m_next_row];
            }
            if (!rows.empty() && m_next_row > 0)
                prune(rows[m_next_row - 1]);
            return factors;
        }

    private:
        struct Reading
        {
            double time;
            gtsam::Vector3 value;
            gtsam::Matrix3 covariance;
        };

        static Types::SensorBase::pStatefulType checked(const std::shared_ptr<PriorT> &prior)
        {
            if (!prior)
                throw std::invalid_argument("AccelerometerSensor requires a non-null prior.");
            return prior;
        }

        /// Side B's placeholders, moved to the 'r' character so they never collide with side A's.
        static typename NodeType::KeyMap sideBKeys()
        {
            typename NodeType::KeyMap map;
            for (const gtsam::Key key : Keys::keys())
                map[key] = gtsam::Symbol('r', gtsam::Symbol(key).index());
            return map;
        }

        void buffer(const pMeasurementType &measurement)
        {
            const auto where = [&]() { return "(t=" + std::to_string(measurement->time) + ", s=" + std::to_string(measurement->arclength) + ")"; };
            if (std::abs(measurement->arclength - m_arclength) > TOLERANCE)
            {
                warning() << "AccelerometerSensor: dropping the sample at " << where() << ": the sensor is at s=" << m_arclength << ".";
                return;
            }
            if (measurement->value.size() != 3 || measurement->covariance.rows() != 3 || measurement->covariance.cols() != 3)
            {
                warning() << "AccelerometerSensor: dropping the sample at " << where() << ": expected a 3x1 value and a 3x3 covariance.";
                return;
            }
            if ((measurement->mask.array() == 0).any())
            {
                warning() << "AccelerometerSensor: dropping the sample at " << where() << ": it has a masked component.";
                return;
            }
            if (m_closed_until && measurement->time < *m_closed_until - TOLERANCE)
            {
                warning() << "AccelerometerSensor: dropping the sample at " << where() << ": its interval's factor was already built (up to t="
                          << *m_closed_until << ").";
                return;
            }
            const Reading reading{measurement->time, gtsam::Vector3(measurement->value), gtsam::Matrix3(measurement->covariance)};
            const auto position = std::upper_bound(m_readings.begin(), m_readings.end(), reading.time,
                                                   [](double t, const Reading &r) { return t < r.time; });
            m_readings.insert(position, reading);
        }

        /// Drops the readings no interval from `boundary` on needs: all but the last one before it.
        void prune(double boundary)
        {
            const auto first_after = std::find_if(m_readings.begin(), m_readings.end(),
                                                  [boundary](const Reading &r) { return r.time >= boundary - TOLERANCE; });
            if (first_after - m_readings.begin() > 1)
                m_readings.erase(m_readings.begin(), first_after - 1);
        }

        std::shared_ptr<gtsam::NonlinearFactor> intervalFactor(double t0, double t1) const
        {
            if (t0 < m_prior->windowStart() - TOLERANCE)
            {
                warning() << "AccelerometerSensor: dropping the interval [" << t0 << ", " << t1 << "]: its first row was already marginalized.";
                return nullptr;
            }

            AccelerometerPreintegration::BiasWeightsFn bias_weights;
            if (m_bias)
                bias_weights = [bias = m_bias](double t) { return bias->weights(t); };
            AccelerometerPreintegration preintegration(t0, t1, std::move(bias_weights));
            for (const Reading &reading : m_readings)
            {
                if (reading.time >= t1 - TOLERANCE)
                    break;
                preintegration.addSample(reading.time, reading.value, reading.covariance);
            }
            preintegration.close();
            if (!preintegration.complete())
            {
                warning() << "AccelerometerSensor: dropping the interval [" << t0 << ", " << t1 << "]: its samples cover only "
                          << preintegration.coveredTime() << " s of it.";
                return nullptr;
            }

            const auto node_a = m_prior->interpolatedNode(t0, m_arclength);
            const auto node_b = m_prior->interpolatedNode(t1, m_arclength, sideBKeys());
            if (!node_a || !node_b)
            {
                warning() << "AccelerometerSensor: dropping the interval [" << t0 << ", " << t1 << "]: it could not be resolved on the prior's nodes.";
                return nullptr;
            }
            auto inner = std::make_shared<AccelerometerFactor>(node_a->innerKey(Keys::Pose()), node_a->innerKey(Keys::Velocity()),
                                                     node_b->innerKey(Keys::Pose()), node_b->innerKey(Keys::Velocity()),
                                                     preintegration, m_options);
            return std::make_shared<FactorType>(inner, *node_a, *node_b);
        }

        std::shared_ptr<PriorT> m_prior;
        double m_arclength;
        AccelerometerOptions m_options;
        std::shared_ptr<BiasModelType> m_bias;

        std::vector<Reading> m_readings;     // buffered, in time order
        size_t m_next_row = 0;               // next row of the prior to close an interval at
        std::optional<double> m_closed_until; // end of the latest interval handled
    };
} // namespace Spacetime::Sensors

#endif // ACCELEROMETERSENSOR_HPP
