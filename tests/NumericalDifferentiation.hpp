#ifndef NUMERICAL_DIFFERENTIATION_HPP
#define NUMERICAL_DIFFERENTIATION_HPP

// Numerical-differentiation testing tool for WNOA::NodeValues<Pose>-valued
// interpolation code (Interpolator, InterpolationFactor). This is a TEST-ONLY
// utility -- it is deliberately kept out of the production headers under
// spacetime/, since central-difference Jacobians are far too slow to use at
// solve time; production code must use analytic Jacobians. Its only purpose is
// to independently validate those analytic Jacobians here in the test suite.
//
// Convention: perturbations and local coordinates use gtsam's own (right/body)
// Retract/Local convention for the pose block -- i.e. pose.retract(v) = pose *
// Expmap(v) and base.localCoordinates(other) = Logmap(base^-1 * other) -- and
// simple vector-space addition/subtraction for the strain/velocity blocks.
// This matches the convention gtsam itself uses for Pose3-valued optimization
// variables, which is what the analytic Jacobians returned to
// InterpolationFactor (and consumed by gtsam's linear solver) must also use.

#include <functional>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/base/Lie.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/prior_models/WNOA/WNOAState.hpp>

namespace Spacetime::Testing
{
    template <typename Pose>
    struct NumericalDifferentiation
    {
        using ValuesType = typename Priors::WNOA::Node<Pose>::ValuesType;
        static constexpr int dim = gtsam::traits<Pose>::dimension;
        static constexpr int node_dim = Priors::WNOA::Node<Pose>::dim;
        using VectorNode = Eigen::Matrix<double, node_dim, 1>;

        /// Retract a node by a node_dim local perturbation (gtsam's own right
        /// convention for the pose block, additive for strain/velocity).
        static ValuesType retractNode(const ValuesType &node, const VectorNode &delta)
        {
            ValuesType result;
            result.pose(gtsam::traits<Pose>::Retract(node.pose(), delta.template segment<dim>(0)));
            result.epsilon(node.epsilon() + delta.template segment<dim>(dim));
            result.varpi(node.varpi() + delta.template segment<dim>(2 * dim));
            return result;
        }

        /// Local coordinates of `other` relative to `base` (gtsam's own right
        /// convention for the pose block, subtraction for strain/velocity).
        static VectorNode localCoordinatesNode(const ValuesType &base, const ValuesType &other)
        {
            VectorNode result;
            result.template segment<dim>(0) = gtsam::traits<Pose>::Local(base.pose(), other.pose());
            result.template segment<dim>(dim) = other.epsilon() - base.epsilon();
            result.template segment<dim>(2 * dim) = other.varpi() - base.varpi();
            return result;
        }

        /**
         * @brief Numerically (central-difference) differentiate a function of a
         *        single ValuesType node (e.g. one argument of interpolateTime /
         *        interpolateSpace bound to a fixed second argument) with respect
         *        to that node's own node_dim local perturbation.
         */
        static Eigen::Matrix<double, node_dim, node_dim> numericalJacobian(
            const std::function<ValuesType(const ValuesType &)> &f,
            const ValuesType &node, double h = 1e-6)
        {
            const ValuesType base = f(node);
            Eigen::Matrix<double, node_dim, node_dim> J;
            for (int i = 0; i < node_dim; ++i)
            {
                VectorNode delta = VectorNode::Zero();
                delta(i) = h;
                const ValuesType plus = f(retractNode(node, delta));
                const ValuesType minus = f(retractNode(node, -delta));
                J.col(i) = (localCoordinatesNode(base, plus) - localCoordinatesNode(base, minus)) / (2 * h);
            }
            return J;
        }

        /**
         * @brief Numerically differentiate a function of a single ValuesType node
         *        with respect to only ONE dim-wide sub-block (0 = pose, 1 =
         *        strain, 2 = velocity) of that node's local perturbation. Used to
         *        validate per-key Jacobian blocks (e.g. Interpolator::interpolate's
         *        jacobians map, keyed by pose/strain/velocity key).
         */
        static Eigen::Matrix<double, node_dim, dim> numericalJacobianBlock(
            const std::function<ValuesType(const ValuesType &)> &f,
            const ValuesType &node, int block, double h = 1e-6)
        {
            const ValuesType base = f(node);
            Eigen::Matrix<double, node_dim, dim> J;
            for (int i = 0; i < dim; ++i)
            {
                VectorNode delta = VectorNode::Zero();
                delta(block * dim + i) = h;
                const ValuesType plus = f(retractNode(node, delta));
                const ValuesType minus = f(retractNode(node, -delta));
                J.col(i) = (localCoordinatesNode(base, plus) - localCoordinatesNode(base, minus)) / (2 * h);
            }
            return J;
        }
    };

    /**
     * @brief Central-difference Jacobians of `factor`'s unwhitened error with respect to
     *        each of its keys, in keys() order, perturbing through gtsam::Values::retract
     *        (so every variable type uses gtsam's own convention).
     */
    inline std::vector<gtsam::Matrix> numericalFactorJacobians(const gtsam::NoiseModelFactor &factor, const gtsam::Values &values, double h = 1e-6)
    {
        std::vector<gtsam::Matrix> H;
        for (const gtsam::Key key : factor.keys())
        {
            const size_t dim = values.at(key).dim();
            gtsam::Matrix J(factor.dim(), dim);
            for (size_t i = 0; i < dim; i++)
            {
                gtsam::VectorValues delta;
                delta.insert(key, gtsam::Vector::Unit(dim, i) * h);
                J.col(i) = (factor.unwhitenedError(values.retract(delta)) - factor.unwhitenedError(values.retract(-1.0 * delta))) / (2 * h);
            }
            H.push_back(J);
        }
        return H;
    }
} // namespace Spacetime::Testing

#endif // NUMERICAL_DIFFERENTIATION_HPP
