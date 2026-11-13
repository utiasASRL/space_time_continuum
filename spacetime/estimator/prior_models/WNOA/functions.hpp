#ifndef WNOAFUNCTIONS_HPP
#define WNOAFUNCTIONS_HPP

#include <vector>
#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/functions.hpp>
#include <spacetime/estimator/prior_models/WNOA/WNOAState.hpp>

namespace Spacetime::Priors::WNOA
{
    /**
     * @brief WNOA transition (Phi) and process-noise (Q) matrices along each axis, over the
     *        node tangent order [pose, strain, velocity], plus node-averaging helpers.
     *
     * In time, the pose integrates velocity (Q1) and strain is a random walk (Q3); in
     * arclength, the pose integrates strain (Q2) and velocity is a random walk (Q4).
     */
    template <typename Pose>
    class Functions
    {
        using NodeType = Node<Pose>;
        static constexpr unsigned int dim = gtsam::traits<Pose>::dimension;
        static constexpr unsigned int node_dim = NodeType::dim;

    public:
        using MatrixN = Eigen::Matrix<double, dim, dim>;
        using Matrix3N = Eigen::Matrix<double, node_dim, node_dim>;
        using VectorN = Eigen::Vector<double, dim>;
        using Vector3N = Eigen::Vector<double, node_dim>;
        using Jacobian = gtsam::OptionalMatrixType;

        /// Accumulated process noise Q_t(dt) over a time step dt > 0.
        static Matrix3N getQt(double dt, const MatrixN &Q1, const MatrixN &Q3)
        {
            assert(dt > 0 && "Time must be positive.");
            const double dt2 = dt * dt / 2.0;
            const double dt3 = dt * dt * dt / 3.0;

            Matrix3N Qt;
            Qt << dt3 * Q1, ZERO(dim), dt2 * Q1,
                ZERO(dim), dt * Q3, ZERO(dim),
                dt2 * Q1, ZERO(dim), dt * Q1;

            return Qt;
        }

        /// Accumulated process noise Q_s(ds) over an arclength step ds > 0.
        static Matrix3N getQs(double ds, const MatrixN &Q2, const MatrixN &Q4)
        {
            assert(ds > 0 && "Arclength must be positive.");
            const double ds2 = ds * ds / 2.0;
            const double ds3 = ds * ds * ds / 3.0;

            Matrix3N Qs;
            Qs << ds3 * Q2, ds2 * Q2, ZERO(dim),
                ds2 * Q2, ds * Q2, ZERO(dim),
                ZERO(dim), ZERO(dim), ds * Q4;

            return Qs;
        }

        /// Closed-form inverse of getQt; Q1 and Q3 must be diagonal.
        static Matrix3N getQtInv(double dt, const MatrixN &Q1, const MatrixN &Q3)
        {
            assert(dt > 0 && "Time must be positive.");
            const MatrixN Q1_inv = invertDiagonal(Q1);
            const MatrixN Q3_inv = invertDiagonal(Q3);

            Matrix3N Qt_inv;
            Qt_inv << 12.0 * Q1_inv / (dt * dt * dt), ZERO(dim), -6.0 * Q1_inv / (dt * dt),
                ZERO(dim), 1.0 * Q3_inv / dt, ZERO(dim),
                -6.0 * Q1_inv / (dt * dt), ZERO(dim), 4.0 * Q1_inv / dt;

            return Qt_inv;
        }

        /// Closed-form inverse of getQs; Q2 and Q4 must be diagonal.
        static Matrix3N getQsInv(double ds, const MatrixN &Q2, const MatrixN &Q4)
        {
            assert(ds > 0 && "Arclength must be positive.");
            const MatrixN Q2_inv = invertDiagonal(Q2);
            const MatrixN Q4_inv = invertDiagonal(Q4);

            // Inverse of the (pose, strain) 2x2 WNOA block (indices 0,1, coupled via Q2) is
            // the standard [[12/d^3, -6/d^2], [-6/d^2, 4/d]] * Q2_inv pattern (matching
            // getQtInv's treatment of its own coupled (pose, velocity) pair at indices
            // 0,2); the decoupled velocity block (index 2) is a plain first-order inverse,
            // Q4_inv/ds. Verified directly against getQs(ds,Q2,Q4) (Qs * QsInv == I) --
            // the previous version here had the wrong sign on the off-diagonal block and
            // had the Q2/Q4 diagonal terms swapped between the (1,1) and (2,2) blocks.
            Matrix3N Qs_inv;
            Qs_inv << 12.0 * Q2_inv / (ds * ds * ds), -6.0 * Q2_inv / (ds * ds), ZERO(dim),
                -6.0 * Q2_inv / (ds * ds), 4.0 * Q2_inv / ds, ZERO(dim),
                ZERO(dim), ZERO(dim), 1.0 * Q4_inv / ds;

            return Qs_inv;
        }

        /// Time transition Phi_t(dt): pose advances by dt * velocity.
        static Matrix3N getPhiT(double dt)
        {
            const double _dt = double(dt);
            Matrix3N Phi_t;
            Phi_t << EYE(dim), ZERO(dim), _dt * EYE(dim),
                ZERO(dim), EYE(dim), ZERO(dim),
                ZERO(dim), ZERO(dim), EYE(dim);
            return Phi_t;
        };

        /// Arclength transition Phi_s(ds): pose advances by ds * strain.
        static Matrix3N getPhiS(double ds)
        {
            const double _ds = double(ds);
            Matrix3N Phi_s;
            Phi_s << EYE(dim), _ds * EYE(dim), ZERO(dim),
                ZERO(dim), EYE(dim), ZERO(dim),
                ZERO(dim), ZERO(dim), EYE(dim);
            return Phi_s;
        }

        // static void globalToLocal(const Values<Pose> &node0, const Values<Pose> &node1, Vector3N &gamma, Jacobian H_0 = Jacobian(), Jacobian H_1 = Jacobian())
        // {
        //     Jacobian dT01_d0, dT01_d1;
        //     gtsam::Pose3 T01 = node0.pose.between(node1.pose, dT01_d0, dT01_d1);
        //     Jacobian dxi_dT01;
        //     gtsam::Vector6 xi = gtsam::traits<gtsam::Pose3>::Logmap(T01, dxi_dT01);

        //     gamma.block<dim, 1>(0, 0) = xi;
        //     gamma.block<dim, 1>(dim, 0) = dxi_dT01 * node1.epsilon;
        //     gamma.block<dim, 1>(dim * 2, 0) = dxi_dT01 * node1.varpi;

        //     if (H_0)
        //     {
        //         *H_0 << (Matrix3N() << dxi_dT01 * dT01_d0, ZERO(dim), ZERO(dim),
        //                  ZERO(dim), ZERO(dim), ZERO(dim),
        //                  ZERO(dim), ZERO(dim), ZERO(dim))
        //                     .finished(); // TODO: Fix.
        //     }
        //     if (H_1)
        //     {
        //         *H_1 << (Matrix3N() << dxi_dT01 * dT01_d1, dxi_dT01, dxi_dT01,
        //                  ZERO(dim), dxi_dT01, ZERO(dim),
        //                  ZERO(dim), ZERO(dim), dxi_dT01)
        //                     .finished(); // TODO: Fix.
        //     }
        // }

        // static void globalToLocal(const Values<Pose> &node, const Values<Pose> &node_nk, Vector3N &gamma, Jacobian dgdx0)
        // {
        //     Matrix3N dump = Matrix3N::Zero();
        //     globalToLocal(node, node_nk, gamma, dgdx0, dump);
        // }

        // static void globalToLocal(const Values<Pose> &node, const Values<Pose> &node_nk, Vector3N &gamma)
        // {
        //     Matrix3N dump = Matrix3N::Zero();
        //     globalToLocal(node, node_nk, gamma, dump, dump);
        // }

        // static void localToGlobal(const Vector3N &gamma, const Values<Pose> &node_nk, Values<Pose> &node)
        // {
        //     // throw std::runtime_error("localToGlobal not implemented.");
        //     gtsam::Vector6 xi = gamma.block<dim, 1>(0, 0);
        //     MatrixN jac;
        //     gtsam::Pose3 tran = gtsam::traits<gtsam::Pose3>::Expmap(xi, jac) * node_nk.pose;
        //     node.pose = tran;
        //     node.epsilon = jac * gamma.block<dim, 1>(dim, 0);
        //     node.varpi = jac * gamma.block<dim, 1>(dim * 2, 0);
        // }

        /// Midpoint of two node values (pose via Pose::interpolateRt, tangents arithmetically); used for initial guesses.
        static Values<Pose> average(const Values<Pose> &values1, const Values<Pose> &values2)
        {
            Values<Pose> return_values;
            return_values.pose(values1.pose().interpolateRt(values2.pose(), 0.5));
            return_values.epsilon(0.5 * (values1.epsilon() + values2.epsilon()));
            return_values.varpi(0.5 * (values1.varpi() + values2.varpi()));
            return return_values;
        }

        /// Midpoint of two nodes' values read from `values`.
        static Values<Pose> average(const gtsam::Values &values, const NodeType &node1, const NodeType &node2)
        {
            Values<Pose> return_values = average(node1.getValues(values), node2.getValues(values));
            return return_values;
        }
    };
}

#endif // WNOAFUNCTIONS_HPP