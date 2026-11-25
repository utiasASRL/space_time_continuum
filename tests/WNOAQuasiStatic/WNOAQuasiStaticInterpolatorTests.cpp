#include <memory>
#include <unordered_map>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic.hpp>

namespace Spacetime::Priors::WNOAQuasiStatic
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Node<Pose>;
        using ValuesType = NodeValues<Pose>;
        using PriorType = Prior<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;
        using Matrix12 = Eigen::Matrix<double, 12, 12>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;

        const Matrix6 Q2 = Vector6(0.6, 0.5, 0.4, 0.3, 0.2, 0.1).asDiagonal();
        const Vector6 epsilon0 = (Vector6() << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).finished();

        /// A 3-node, unit-length robot with two rows (t = 0 and t = 1), each set to a
        /// different bent shape.
        struct Fixture
        {
            KeyManager km;
            std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, 3);
            std::shared_ptr<PriorType> prior = std::make_shared<PriorType>(state, ValuesType(Pose(), epsilon0), Matrix12::Identity(), Q2);
            gtsam::Values values;

            Fixture()
            {
                for (const double t : {0.0, 1.0})
                {
                    gtsam::Values new_values;
                    prior->extendState(km, t, values, new_values);
                    values.insert(new_values);
                }
                for (unsigned int row = 0; row < 2; row++)
                    for (unsigned int n = 0; n < 3; n++)
                    {
                        values.update(node(row, n)->poseKey(), shape(row, n).pose());
                        values.update(node(row, n)->strainKey(), shape(row, n).epsilon());
                    }
            }

            static ValuesType shape(unsigned int row, unsigned int n)
            {
                const Vector6 strain = (Vector6() << 0.1 * (row + 1), -0.2 * n, 0.05, 0.01 * n, 0.0, 1.0).finished();
                return ValuesType(Pose::Expmap((0.5 * n) * strain), strain);
            }

            std::shared_ptr<NodeType> node(unsigned int row, unsigned int n) const { return state->estimation_nodes.at(row * 3 + n); }

            std::shared_ptr<ValuesType> query(double time, double arclength, Eigen::MatrixXd *cov = nullptr) const
            {
                return std::dynamic_pointer_cast<ValuesType>(prior->query(time, arclength, values, cov));
            }
        };

        /// The same node values as a WNOA node, with a zero velocity.
        WNOA::NodeValues<Pose> asWNOA(const ValuesType &v)
        {
            WNOA::NodeValues<Pose> out;
            out.pose(v.pose());
            out.epsilon(v.epsilon());
            return out;
        }
    }

    TEST(WNOAQuasiStaticInterpolator, ArclengthInterpolationIsTheWNOAInterpolatorsPoseStrainSlice)
    {
        // In arclength, WNOA's (pose, strain) block never depends on the velocity, so the
        // quasi-static interpolation is exactly that block of WNOA::Interpolator's.
        Fixture f;
        const WNOA::Interpolator<Pose> wnoa(Matrix6::Identity(), Q2, Matrix6::Identity(), Matrix6::Identity());
        for (const double s : {0.1, 0.25, 0.4})
        {
            Eigen::MatrixXd cov;
            const auto queried = f.query(0.0, s, &cov);
            ASSERT_TRUE(queried);

            Eigen::Matrix<double, 18, 18> wnoa_cov;
            const WNOA::NodeValues<Pose> expected = wnoa.interpolateSpace(
                0.0, 0.5, s, asWNOA(Fixture::shape(0, 0)), asWNOA(Fixture::shape(0, 1)), nullptr, &wnoa_cov);
            EXPECT_TRUE(queried->pose().equals(expected.pose(), 1e-12)) << "s = " << s;
            EXPECT_TRUE(queried->epsilon().isApprox(expected.epsilon(), 1e-12)) << "s = " << s;
            ASSERT_EQ(cov.rows(), 12);
            EXPECT_TRUE(cov.isApprox(wnoa_cov.topLeftCorner<12, 12>(), 1e-12)) << "s = " << s;
            EXPECT_GT(cov.trace(), 0.0);
        }
    }

    TEST(WNOAQuasiStaticInterpolator, ArclengthJacobianIsTheWNOAInterpolatorsPoseStrainSlice)
    {
        const WNOA::Interpolator<Pose> wnoa(Matrix6::Identity(), Q2, Matrix6::Identity(), Matrix6::Identity());
        const Interpolator<Pose> interpolator(Q2);
        gtsam::Matrix wnoa_jac, jac;
        wnoa.interpolateSpace(0.0, 0.5, 0.2, asWNOA(Fixture::shape(0, 1)), asWNOA(Fixture::shape(0, 2)), &wnoa_jac);
        interpolator.interpolateSpace(0.0, 0.5, 0.2, Fixture::shape(0, 1), Fixture::shape(0, 2), &jac);
        ASSERT_EQ(jac.rows(), 12);
        ASSERT_EQ(jac.cols(), 24);
        EXPECT_TRUE(jac.leftCols<12>().isApprox(wnoa_jac.block(0, 0, 12, 12), 1e-12));
        EXPECT_TRUE(jac.rightCols<12>().isApprox(wnoa_jac.block(0, 18, 12, 12), 1e-12));
    }

    TEST(WNOAQuasiStaticInterpolator, TimeQueriesSnapToTheNearestRow)
    {
        Fixture f;
        for (const double s : {0.0, 0.5, 0.3})
        {
            const auto row0 = f.query(0.0, s), row1 = f.query(1.0, s);
            for (const double t : {0.2, 0.49})
            {
                Eigen::MatrixXd cov;
                const auto snapped = f.query(t, s, &cov);
                ASSERT_TRUE(snapped);
                EXPECT_TRUE(snapped->pose().equals(row0->pose(), 1e-12)) << "t = " << t << ", s = " << s;
                EXPECT_TRUE(snapped->epsilon().isApprox(row0->epsilon(), 1e-12));
            }
            for (const double t : {0.51, 0.9})
            {
                const auto snapped = f.query(t, s);
                ASSERT_TRUE(snapped);
                EXPECT_TRUE(snapped->pose().equals(row1->pose(), 1e-12)) << "t = " << t << ", s = " << s;
            }
        }
    }

    TEST(WNOAQuasiStaticInterpolator, SnappingInTimeAddsNoConditionalCovariance)
    {
        Fixture f;
        Eigen::MatrixXd on_node, snapped_node, on_row, snapped_row;
        f.query(1.0, 0.5, &on_node);
        f.query(0.8, 0.5, &snapped_node);
        EXPECT_TRUE(on_node.isZero());
        EXPECT_TRUE(snapped_node.isZero());

        f.query(1.0, 0.3, &on_row);
        f.query(0.8, 0.3, &snapped_row);
        EXPECT_TRUE(snapped_row.isApprox(on_row, 1e-12));
    }

    TEST(WNOAQuasiStaticInterpolator, SnappedJacobiansDependOnTheNearestRowOnly)
    {
        Fixture f;
        std::unordered_map<gtsam::Key, Eigen::MatrixXd> jacobians;
        f.prior->interpolator().interpolate(0.8, 0.0, f.values, f.node(0, 0), f.node(1, 0), nullptr, nullptr, &jacobians);
        EXPECT_TRUE(jacobians.at(f.node(0, 0)->poseKey()).isZero());
        EXPECT_TRUE(jacobians.at(f.node(0, 0)->strainKey()).isZero());
        EXPECT_TRUE(jacobians.at(f.node(1, 0)->poseKey()).topRows<6>().isIdentity());
        EXPECT_TRUE(jacobians.at(f.node(1, 0)->strainKey()).bottomRows<6>().isIdentity());
    }

    TEST(WNOAQuasiStaticInterpolator, QueriesAfterTheLastRowHoldIt)
    {
        Fixture f;
        Eigen::MatrixXd cov;
        const auto held = f.query(5.0, 0.5, &cov);
        ASSERT_TRUE(held);
        EXPECT_TRUE(held->pose().equals(Fixture::shape(1, 1).pose(), 1e-12));
        EXPECT_TRUE(cov.isZero());
    }

    TEST(WNOAQuasiStaticInterpolator, QueriesBeforeTheFirstRowAreUnavailable)
    {
        Fixture f;
        EXPECT_EQ(f.query(-0.1, 0.5), nullptr);
    }

    TEST(WNOAQuasiStaticInterpolator, ArclengthExtrapolationFollowsTheStrain)
    {
        // Past the tip, the WNOA (pose, strain) extrapolation is the constant-strain arc.
        auto state = std::make_shared<StateType>(1.0, 3);
        KeyManager km;
        PriorType prior(state, ValuesType(Pose(), epsilon0), Matrix12::Identity(), Q2);
        gtsam::Values values, new_values;
        prior.extendState(km, 0.0, values, new_values);
        values.insert(new_values);

        const auto extrapolated = std::dynamic_pointer_cast<ValuesType>(prior.query(0.0, 1.5, values));
        ASSERT_TRUE(extrapolated);
        EXPECT_TRUE(extrapolated->pose().equals(Pose::Expmap(1.5 * epsilon0), 1e-12));
        EXPECT_TRUE(extrapolated->epsilon().isApprox(epsilon0));
    }
} // namespace Spacetime::Priors::WNOAQuasiStatic
