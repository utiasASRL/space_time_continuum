#include <array>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/Values.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using Vector6 = Eigen::Matrix<double, 6, 1>;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;

        /// A 2 x 2 grid of nodes (s in {0, 1}, t in {0, 1}) with small, distinct values.
        template <typename NodeT>
        struct Grid
        {
            KeyManager km;
            std::shared_ptr<NodeT> n00, n01, n10, n11;

            Grid()
                : n00(std::make_shared<NodeT>(km, 0.0, 0.0)), n01(std::make_shared<NodeT>(km, 0.0, 1.0)),
                  n10(std::make_shared<NodeT>(km, 1.0, 0.0)), n11(std::make_shared<NodeT>(km, 1.0, 1.0)) {}

            /// Values of every corner, varied by `seed` so different calls give different states.
            gtsam::Values values(int seed) const
            {
                gtsam::Values out;
                int i = 0;
                for (const auto &node : {n00, n01, n10, n11})
                {
                    typename NodeT::ValuesType v;
                    v.setRandom();
                    Vector6 xi;
                    for (int k = 0; k < 6; k++)
                        xi(k) = 0.02 * std::sin(1.0 + seed + 7 * i + 3 * k);
                    v.pose(Pose::Expmap(xi));
                    if constexpr (requires { v.epsilon(Vector6()); })
                    {
                        v.epsilon((Vector6() << 0, 0, 0, 1, 0, 0).finished() + 0.5 * xi);
                        v.varpi(0.3 * xi);
                    }
                    node->setValues(out, v);
                    i++;
                }
                return out;
            }
        };

        struct Query
        {
            const char *name;
            double time, arclength;
            bool use01, use10; // node11 is used when both are
        };

        // Every shape interpolateCorners dispatches on.
        const std::array<Query, 7> kQueries = {{
            {"coincide", 0.0, 0.0, true, true},
            {"time only", 0.4, 0.0, true, false},
            {"space only", 0.0, 0.7, false, true},
            {"both interpolate", 0.4, 0.7, true, true},
            {"time interpolate, space extrapolate", 0.4, 0.3, true, false},
            {"space interpolate, time extrapolate", 0.4, 0.7, false, true},
            {"both extrapolate", 0.4, 0.3, false, false},
        }};

        /// Interpolates every query with a plan made once, against several states, and checks
        /// it against interpolating the query afresh each time.
        template <typename InterpolatorT, typename NodeT>
        void expectPlansMatchFreshQueries(const InterpolatorT &interp)
        {
            Grid<NodeT> grid;
            for (const Query &q : kQueries)
            {
                const auto n01 = q.use01 ? grid.n01 : nullptr;
                const auto n10 = q.use10 ? grid.n10 : nullptr;
                const auto n11 = (q.use01 && q.use10) ? grid.n11 : nullptr;
                const auto plan = interp.makePlan(q.time, q.arclength, grid.n00, n01, n10, n11);

                for (int seed : {0, 1, 2})
                {
                    const gtsam::Values values = grid.values(seed);
                    std::unordered_map<gtsam::Key, Eigen::MatrixXd> planned_jac, fresh_jac;
                    Eigen::MatrixXd planned_cov, fresh_cov;
                    auto planned = std::dynamic_pointer_cast<typename NodeT::ValuesType>(
                        interp.interpolate(plan, values, grid.n00, n01, n10, n11, &planned_jac, &planned_cov));
                    auto fresh = std::dynamic_pointer_cast<typename NodeT::ValuesType>(
                        interp.interpolate(q.time, q.arclength, values, grid.n00, n01, n10, n11, &fresh_jac, &fresh_cov));
                    ASSERT_TRUE(planned && fresh) << q.name;

                    EXPECT_TRUE(planned->pose().equals(fresh->pose(), 1e-12)) << q.name << " seed " << seed;
                    for (size_t k = 0; k < planned->values.size(); k++)
                        EXPECT_TRUE(planned->values[k].isApprox(fresh->values[k], 1e-12) ||
                                    (planned->values[k] - fresh->values[k]).norm() < 1e-12)
                            << q.name << " component " << k;
                    EXPECT_LT((planned_cov - fresh_cov).norm(), 1e-12) << q.name;
                    ASSERT_EQ(planned_jac.size(), fresh_jac.size()) << q.name;
                    for (const auto &[key, block] : fresh_jac)
                        EXPECT_LT((planned_jac.at(key) - block).norm(), 1e-12) << q.name;
                }
            }
        }
    } // namespace

    TEST(InterpolationPlan, WNOAPlansMatchFreshQueriesForEveryShape)
    {
        using Interp = Priors::WNOA::Interpolator<Pose>;
        expectPlansMatchFreshQueries<Interp, Priors::WNOA::Node<Pose>>(
            Interp(Matrix6::Identity() * 0.7, Matrix6::Identity() * 0.5, Matrix6::Identity() * 0.3, Matrix6::Identity() * 0.2));
    }

    TEST(InterpolationPlan, WNOVPlansMatchFreshQueriesForEveryShape)
    {
        using Interp = Priors::WNOV::Interpolator<Pose>;
        expectPlansMatchFreshQueries<Interp, Priors::WNOV::Node<Pose>>(
            Interp(Vector6(1, 2, 3, 4, 5, 6).asDiagonal(), Vector6(0.6, 0.5, 0.4, 0.3, 0.2, 0.1).asDiagonal(),
                   (Vector6() << 0, 0.3, 0, 0, 0, 1).finished()));
    }

    TEST(InterpolationPlan, InvalidQueriesAreRejectedWhenPlanned)
    {
        using Interp = Priors::WNOA::Interpolator<Pose>;
        const Interp interp(Matrix6::Identity(), Matrix6::Identity(), Matrix6::Identity(), Matrix6::Identity());
        Grid<Priors::WNOA::Node<Pose>> grid;
        EXPECT_THROW(interp.makePlan(0.4, 0.7, nullptr, grid.n01, grid.n10, grid.n11), std::runtime_error);
        EXPECT_THROW(interp.makePlan(0.4, 0.7, grid.n10, grid.n01, grid.n00, nullptr), std::runtime_error); // node10 behind node00
        EXPECT_THROW(interp.makePlan(-0.1, 0.0, grid.n00, grid.n01, nullptr, nullptr), std::runtime_error); // before node00
        EXPECT_THROW(interp.makePlan(1.5, 0.0, grid.n00, grid.n01, nullptr, nullptr), std::runtime_error);  // past node01
        EXPECT_THROW(interp.makePlan(0.4, 0.7, grid.n00, grid.n01, grid.n10, nullptr), std::runtime_error); // node11 missing
    }
} // namespace Spacetime
