#ifndef BIAS_HPP
#define BIAS_HPP

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/base/Lie.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>

#include <spacetime/estimator/types/ActuatorBase.hpp>
#include <spacetime/estimator/types/ContinuumRobot.hpp>
#include <spacetime/estimator/types/InterpolatedKeys.hpp>
#include <spacetime/estimator/types/Robot.hpp>
#include <spacetime/estimator/types/SensorBase.hpp>
#include <spacetime/estimator/PosteriorCovariance.hpp>
#include <spacetime/estimator/utilities/functions.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

namespace Spacetime::Biases
{
    template <typename T>
    struct BiasKeys;

    /**
     * @brief Values of a bias node: the bias, of any gtsam manifold type T (e.g. a
     *        gtsam::Vector3 accelerometer bias, or a gtsam::Pose3 sensor extrinsic). Stored
     *        in Types::NodeValuesBase::values as its matrix.
     * @tparam T The bias type.
     */
    template <typename T>
    struct BiasNodeValues : public Types::NodeValuesBase
    {
        using BiasType = T;                                              ///< The bias type.
        using InterpolatedKeys = BiasKeys<T>;                            ///< The bias's interpolated key map.
        static constexpr unsigned int dim = gtsam::traits<T>::dimension; ///< Tangent dimension of T.

        /// The identity bias.
        BiasNodeValues() : BiasNodeValues(gtsam::traits<T>::Identity()) {}

        /// Holds `value`.
        explicit BiasNodeValues(const T &value)
        {
            values.resize(1);
            bias(value);
        }

        /// The bias.
        T bias() const { return T(values.at(0)); }
        /// Sets the bias.
        void bias(const T &value) { values[0] = value.matrix(); }
    };

    /// A bias's interpolated key (see Types::InterpolatedKeyMap): the whole node. @tparam T The bias type.
    template <typename T>
    struct BiasKeys
    {
        using ValuesType = BiasNodeValues<T>;                       ///< The node-values type.
        static constexpr int dim = gtsam::traits<T>::dimension; ///< Tangent dimension of T.

        /// Interpolated bias key. Only needs to be distinct from the other keys of an inner factor.
        static gtsam::Key Bias() { return gtsam::Symbol('B', 0); }
        /// The single interpolated key, Bias().
        static gtsam::KeyVector keys() { return {Bias()}; }

        /// Inserts `node`'s bias into `values` at `key`. @throws std::runtime_error unless `key` is Bias().
        static void insert(gtsam::Key key, const ValuesType &node, gtsam::Values &values)
        {
            if (key != Bias())
                throw std::runtime_error("BiasKeys::insert: key is not the bias's interpolated key.");
            values.insert(key, node.bias());
        }

        /// (0, dim): the bias spans the whole node. @throws std::runtime_error unless `key` is Bias().
        static std::pair<int, int> rowRange(gtsam::Key key)
        {
            if (key != Bias())
                throw std::runtime_error("BiasKeys::rowRange: key is not the bias's interpolated key.");
            return {0, dim};
        }
    };

    /**
     * @brief A bias node at one (arclength, time): a single "bias" variable, keyed with the
     *        'b' character (KeyManager numbers keys per character, so several bias models
     *        share it without collisions).
     * @tparam T The bias type.
     */
    template <typename T>
    struct BiasNode : public Types::NodeBase
    {
        using ValuesType = BiasNodeValues<T>;                                ///< The node-values type.
        static constexpr unsigned int dim = gtsam::traits<T>::dimension; ///< Tangent dimension of T.

        /// Registers the bias variable with `km`, timestamped at `time`.
        BiasNode(KeyManager &km, double arclength, double time) : Types::NodeBase(km, arclength, time)
        {
            m_bias_key = this->registerVariable<T>(km, "bias", 'b');
        }

        /// Key of the bias variable (of type T).
        gtsam::Key biasKey() const { return m_bias_key; }

        /// The single pair (BiasKeys<T>::Bias(), biasKey()).
        std::vector<std::pair<gtsam::Key, gtsam::Key>> interpolatedKeys() const override { return {{BiasKeys<T>::Bias(), m_bias_key}}; }

        /// The bias read from `values` (gtsam's error if absent).
        std::shared_ptr<Types::NodeValuesBase> getValuesBase(const gtsam::Values &values) const override
        {
            return std::make_shared<ValuesType>(values.at<T>(m_bias_key));
        }

    private:
        gtsam::Key m_bias_key;
    };

    /// The state of a bias: one node per row, at a single arclength. @tparam T The bias type.
    template <typename T>
    using BiasState = Types::ContinuumRobotState<BiasNode<T>>;

    /**
     * @brief Linear interpolation of a bias in time between two bias nodes: the geodesic
     *        b0 * Exp(tau * Log(b0^-1 b1)), which for a vector space is (1 - tau) b0 + tau b1.
     * @tparam T The bias type.
     */
    template <typename T>
    struct BiasInterpolator
    {
        using Jacobian = typename gtsam::MakeJacobian<T, T>::type; ///< dim x dim Jacobian of T w.r.t. T.

        /// The bias a fraction `tau` of the way from `b0` to `b1`, with its Jacobians w.r.t. both.
        static T interpolate(const T &b0, const T &b1, double tau, gtsam::Matrix *H0 = nullptr, gtsam::Matrix *H1 = nullptr)
        {
            if (!H0 && !H1)
                return gtsam::interpolate<T>(b0, b1, tau);
            Jacobian J0, J1;
            const T result = gtsam::interpolate<T>(b0, b1, tau, J0, J1);
            if (H0)
                *H0 = J0;
            if (H1)
                *H1 = J1;
            return result;
        }
    };

    /// The parameters of a BiasModel. The covariances are dim x dim, in T's tangent space. @tparam T The bias type.
    template <typename T>
    struct BiasOptions
    {
        static constexpr int dim = gtsam::traits<T>::dimension; ///< Tangent dimension of T.
        using Matrix = Eigen::Matrix<double, dim, dim>;         ///< A dim x dim covariance.

        /// Prior mean of the first bias node, and every node's initial guess until one is estimated.
        T initial = gtsam::traits<T>::Identity();
        /// Prior covariance of the first bias node.
        Matrix initial_covariance = Matrix::Identity();
        /// Power spectral density Q_c of the bias's random walk: consecutive nodes dt apart differ by N(0, Q_c dt).
        Matrix random_walk_psd = Matrix::Identity();
    };

    /**
     * @class BiasModel
     * @brief A slowly varying bias of type T (e.g. an accelerometer's bias, a gyroscope's,
     *        or a camera's SE(3) extrinsic) estimated as a random walk in time: one bias node
     *        per estimation time, a prior on the first and a BetweenFactor (covariance Q_c dt)
     *        between consecutive ones. Between nodes the bias is interpolated linearly (see
     *        BiasInterpolator), which a sensor folding the bias into its own factor can use
     *        through bracket / weights.
     *
     * A sensor that measures the bias at a single time can use the default
     * StatefulModel::getOuterFactor on BiasKeys::Bias(), which snaps to the nearest node.
     *
     * @tparam T The bias type: any gtsam manifold with gtsam::traits (e.g. gtsam::Vector3, gtsam::Pose3).
     */
    template <typename T>
    class BiasModel : public Types::StatefulModel<BiasState<T>>
    {
    public:
        using BiasType = T;                                     ///< The bias type.
        using NodeType = BiasNode<T>;                           ///< A bias node.
        using ValuesType = BiasNodeValues<T>;                       ///< A bias node's values.
        using StateType = BiasState<T>;                         ///< The bias's state.
        using Options = BiasOptions<T>;                         ///< The model's parameters.
        static constexpr int dim = gtsam::traits<T>::dimension; ///< Tangent dimension of T.

        /// The node(s) bordering a query time and its fraction `tau` of the way between them.
        struct Bracket
        {
            std::shared_ptr<NodeType> node0; ///< The last node at or before the query time.
            /// nullptr when the query is on node0, or past the last node (where the bias is held).
            std::shared_ptr<NodeType> node1;
            double tau = 0.0; ///< Fraction of the way from node0 to node1 (0 without node1).
        };

        /// Weights of each bias node key in a linearly interpolated bias.
        using Weights = std::vector<std::pair<gtsam::Key, double>>;

        /**
         * @param options See BiasOptions.
         * @param arclength The arclength of the bias's nodes (e.g. the sensor's), at which the
         *        default getOuterFactor resolves queries.
         */
        explicit BiasModel(const Options &options = Options(), double arclength = 0.0)
            : Types::StatefulModel<StateType>(std::make_shared<StateType>(std::vector<double>{arclength})), m_options(options) {}

        /// The options the model was built with.
        const Options &options() const { return m_options; }

        /**
         * @brief Adds a bias node at `time` (unless one exists there or later), with its prior
         *        (the first node, initialized at BiasOptions::initial) or random-walk
         *        BetweenFactor (initialized at the previous node's current estimate).
         */
        Types::Model::Factors extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override
        {
            Types::Model::Factors factors;
            const auto state = this->state();
            if (!state->estimation_nodes.empty() && time <= state->estimation_nodes.back()->time + TOLERANCE)
                return factors;

            const std::shared_ptr<NodeType> previous = state->estimation_nodes.empty() ? nullptr : state->estimation_nodes.back();
            state->addStateAtTime(km, time);
            const std::shared_ptr<NodeType> &node = state->estimation_nodes.back();

            if (!previous)
            {
                new_values.insert(node->biasKey(), m_options.initial);
                factors.push_back(std::make_shared<gtsam::PriorFactor<T>>(
                    node->biasKey(), m_options.initial, gtsam::noiseModel::Gaussian::Covariance(m_options.initial_covariance)));
                return factors;
            }

            const gtsam::Key previous_key = previous->biasKey();
            const T guess = current_values.exists(previous_key) ? current_values.at<T>(previous_key)
                            : new_values.exists(previous_key)   ? new_values.at<T>(previous_key)
                                                                : m_options.initial;
            new_values.insert(node->biasKey(), guess);
            factors.push_back(std::make_shared<gtsam::BetweenFactor<T>>(
                previous_key, node->biasKey(), gtsam::traits<T>::Identity(),
                gtsam::noiseModel::Gaussian::Covariance((time - previous->time) * m_options.random_walk_psd)));
            return factors;
        }

        /// The node(s) bordering `time` [s]; nullopt before the first node or without nodes. Past the last node, that node alone (the bias is held).
        std::optional<Bracket> bracket(double time) const
        {
            const auto &nodes = this->state()->estimation_nodes;
            if (nodes.empty() || time < nodes.front()->time - TOLERANCE)
                return std::nullopt;
            // The last node at or before `time`.
            const auto after = std::upper_bound(nodes.begin(), nodes.end(), time + TOLERANCE,
                                                [](double t, const std::shared_ptr<NodeType> &node) { return t < node->time; });
            const std::shared_ptr<NodeType> &node0 = *std::prev(after);
            if (after == nodes.end() || std::abs(time - node0->time) < TOLERANCE)
                return Bracket{node0, nullptr, 0.0};
            const std::shared_ptr<NodeType> &node1 = *after;
            return Bracket{node0, node1, (time - node0->time) / (node1->time - node0->time)};
        }

        /// The interpolation weight of each bias node key at `time` (see bracket); empty if unresolved.
        Weights weights(double time) const
        {
            const auto b = bracket(time);
            if (!b)
                return {};
            if (!b->node1)
                return {{b->node0->biasKey(), 1.0}};
            return {{b->node0->biasKey(), 1.0 - b->tau}, {b->node1->biasKey(), b->tau}};
        }

        /**
         * @brief The bias at `time` (`arclength` is unused), linearly interpolated between its
         *        bordering nodes (see StatefulModel::query). The conditional covariance is the
         *        random walk's, conditioned on both nodes: tau (1 - tau) dt Q_c.
         */
        std::shared_ptr<Types::NodeValuesBase> query(
            double time,
            double arclength,
            const gtsam::Values &values,
            Eigen::MatrixXd *conditionalCovariance = nullptr,
            const PosteriorCovariance *posterior = nullptr,
            Eigen::MatrixXd *posteriorCovariance = nullptr) const override
        {
            (void)arclength;
            const auto b = bracket(time);
            if (!b || !values.exists(b->node0->biasKey()) || (b->node1 && !values.exists(b->node1->biasKey())))
                return nullptr;

            gtsam::KeyVector keys = {b->node0->biasKey()};
            Eigen::MatrixXd J;
            T bias = values.at<T>(b->node0->biasKey());
            Eigen::MatrixXd conditional = Eigen::MatrixXd::Zero(dim, dim);
            if (b->node1)
            {
                keys.push_back(b->node1->biasKey());
                gtsam::Matrix H0, H1;
                bias = BiasInterpolator<T>::interpolate(bias, values.at<T>(b->node1->biasKey()), b->tau, &H0, &H1);
                J.resize(dim, 2 * dim);
                J << H0, H1;
                conditional = b->tau * (1.0 - b->tau) * (b->node1->time - b->node0->time) * m_options.random_walk_psd;
            }
            else
                J = Eigen::MatrixXd::Identity(dim, dim);

            if (conditionalCovariance)
                *conditionalCovariance = conditional;
            if (posteriorCovariance)
            {
                Eigen::MatrixXd joint;
                if (posterior && posterior->joint(keys, joint))
                    *posteriorCovariance = conditional + J * joint * J.transpose();
                else
                    posteriorCovariance->resize(0, 0);
            }
            return std::make_shared<ValuesType>(bias);
        }

        /// Each bias node still in `values`, on its own.
        std::vector<gtsam::KeyVector> covarianceKeyGroups(const gtsam::Values &values) const override
        {
            std::vector<gtsam::KeyVector> groups;
            for (const auto &node : this->state()->estimation_nodes)
                if (values.exists(node->biasKey()))
                    groups.push_back({node->biasKey()});
            return groups;
        }

    private:
        Options m_options;
    };
} // namespace Spacetime::Biases

#endif // BIAS_HPP
