#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/utilities/RobotShape.hpp>

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::NodeValues<Pose>;
        using PriorT = Priors::WNOA::Prior<Pose>;
        using Shape = Utilities::RobotShape<NodeType>;

        constexpr unsigned int kSpatialDim = 4;
        constexpr double kLength = 1.0;

        /// A straight robot along z (unit strain), extended row by row as Estimator::estimate does.
        struct ShapeRig
        {
            std::shared_ptr<StateType> state = std::make_shared<StateType>(kLength, kSpatialDim);
            std::shared_ptr<PriorT> prior;
            gtsam::Values values;
            KeyManager km;
            Shape shape{state};

            ShapeRig()
            {
                ValueType reference;
                reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
                prior = std::make_shared<PriorT>(state, reference);
            }

            void addRow(double time)
            {
                gtsam::Values new_values;
                prior->extendState(km, time, values, new_values);
                values.insert(new_values);
            }

            void offsetRow(unsigned int row, gtsam::Values &target, double dz) const
            {
                for (unsigned int i = 0; i < kSpatialDim; i++)
                {
                    const auto &node = state->estimation_nodes.at(row * kSpatialDim + i);
                    const Pose pose = target.at<Pose>(node->poseKey());
                    target.update(node->poseKey(), Pose(pose.rotation(), pose.translation() + gtsam::Point3(0, 0, dz)));
                }
            }

            Pose nodePose(unsigned int row, unsigned int i, const gtsam::Values &source) const
            {
                return source.at<Pose>(state->estimation_nodes.at(row * kSpatialDim + i)->poseKey());
            }

            static EstimatorSnapshot snapshot(double time, const gtsam::Values &values) { return EstimatorSnapshot{time, values}; }
        };
    } // namespace

    TEST(RobotShape, RejectsNullState)
    {
        EXPECT_THROW(Shape(nullptr), std::invalid_argument);
    }

    TEST(RobotShape, BindPriorInterpolatorRejectsNullOrForeignPrior)
    {
        ShapeRig rig;
        EXPECT_THROW(rig.shape.bindPriorInterpolator(std::shared_ptr<PriorT>()), std::invalid_argument);

        ShapeRig other;
        EXPECT_THROW(rig.shape.bindPriorInterpolator(other.prior), std::invalid_argument);
        EXPECT_FALSE(rig.shape.hasInterpolator());

        rig.shape.bindPriorInterpolator(rig.prior);
        EXPECT_TRUE(rig.shape.hasInterpolator());
    }

    TEST(RobotShape, PoseAtWithoutInterpolatorIsNullopt)
    {
        ShapeRig rig;
        rig.addRow(0.0);
        EXPECT_FALSE(rig.shape.poseAt(0.0, 0.5, ShapeRig::snapshot(0.0, rig.values), rig.state->snapshotEstimationNodes(), false));
    }

    TEST(RobotShape, PoseAtNodeArclengthIsTheNodePose)
    {
        ShapeRig rig;
        rig.shape.bindPriorInterpolator(rig.prior);
        rig.addRow(0.0);
        const auto nodes = rig.state->snapshotEstimationNodes();

        for (unsigned int i = 0; i < kSpatialDim; i++)
        {
            const auto pose = rig.shape.poseAt(0.0, nodes[i]->arclength, ShapeRig::snapshot(0.0, rig.values), nodes, false);
            ASSERT_TRUE(pose) << "node " << i;
            EXPECT_TRUE(Pose(*pose).equals(rig.nodePose(0, i, rig.values), 1e-9)) << "node " << i;
        }
    }

    TEST(RobotShape, PoseAtBetweenNodesInterpolatesAlongTheBackbone)
    {
        ShapeRig rig;
        rig.shape.bindPriorInterpolator(rig.prior);
        rig.addRow(0.0);

        const auto pose = rig.shape.poseAt(0.0, 0.5, ShapeRig::snapshot(0.0, rig.values), rig.state->snapshotEstimationNodes(), false);

        ASSERT_TRUE(pose);
        EXPECT_NEAR((*pose)(2, 3), 0.5, 1e-9); // straight along z with unit strain
    }

    TEST(RobotShape, PoseAtIsNulloptBeforeTheRobotOrItsFirstRow)
    {
        ShapeRig rig;
        rig.shape.bindPriorInterpolator(rig.prior);
        rig.addRow(10.0);
        const auto snapshot = ShapeRig::snapshot(10.0, rig.values);
        const auto nodes = rig.state->snapshotEstimationNodes();

        EXPECT_FALSE(rig.shape.poseAt(10.0, -0.1, snapshot, nodes, false));
        EXPECT_FALSE(rig.shape.poseAt(9.0, 0.5, snapshot, nodes, false));
    }

    TEST(RobotShape, PoseAtIsNulloptWhenABorderingNodeIsMissingFromTheSnapshot)
    {
        ShapeRig rig;
        rig.shape.bindPriorInterpolator(rig.prior);
        rig.addRow(0.0);

        EXPECT_FALSE(rig.shape.poseAt(0.0, 0.5, ShapeRig::snapshot(0.0, gtsam::Values()), rig.state->snapshotEstimationNodes(), false));
    }

    TEST(RobotShape, BackboneHoldsEveryNodeAndTheExtraPointsBetweenThem)
    {
        ShapeRig rig;
        rig.shape.bindPriorInterpolator(rig.prior);
        rig.addRow(0.0);
        constexpr unsigned int extra_points = 2;

        const auto backbone = rig.shape.backbone(0.0, ShapeRig::snapshot(0.0, rig.values), rig.state->snapshotEstimationNodes(), extra_points, false);

        ASSERT_TRUE(backbone);
        EXPECT_DOUBLE_EQ(backbone->row_time, 0.0);
        EXPECT_EQ(backbone->points_per_segment, extra_points + 1);
        ASSERT_EQ(backbone->points.size(), kSpatialDim + (kSpatialDim - 1) * extra_points);
        for (unsigned int i = 0; i < kSpatialDim; i++)
        {
            const auto &point = backbone->points[i * backbone->points_per_segment];
            EXPECT_TRUE(Pose(point.pose).equals(rig.nodePose(0, i, rig.values), 1e-9)) << "node " << i;
        }
        for (std::size_t k = 0; k < backbone->points.size(); k++)
        {
            const double expected_s = kLength * static_cast<double>(k) / (backbone->points.size() - 1);
            EXPECT_NEAR(backbone->points[k].arclength, expected_s, 1e-9) << "point " << k;
            EXPECT_NEAR(backbone->points[k].pose(2, 3), expected_s, 1e-9) << "point " << k;
            EXPECT_EQ(backbone->points[k].covariance.size(), 0) << "point " << k; // no posterior requested
        }
    }

    // Online, estimate(t) appends its row before the snapshot holding it is published.
    TEST(RobotShape, BackboneUsesTheLatestRowTheSnapshotHolds)
    {
        ShapeRig rig;
        rig.shape.bindPriorInterpolator(rig.prior);
        rig.addRow(10.0);
        const gtsam::Values snapshot_values = rig.values;
        rig.addRow(10.1);

        const auto backbone = rig.shape.backbone(10.15, ShapeRig::snapshot(10.0, snapshot_values), rig.state->snapshotEstimationNodes(), 0, false);

        ASSERT_TRUE(backbone);
        EXPECT_DOUBLE_EQ(backbone->row_time, 10.0);
    }

    TEST(RobotShape, BackboneSelectsTheRowAtOrBeforeTheRequestedTime)
    {
        ShapeRig rig;
        rig.shape.bindPriorInterpolator(rig.prior);
        rig.addRow(10.0);
        rig.addRow(10.1);
        gtsam::Values snapshot_values = rig.values;
        rig.offsetRow(1, snapshot_values, 0.5);
        const auto snapshot = ShapeRig::snapshot(10.1, snapshot_values);
        const auto nodes = rig.state->snapshotEstimationNodes();

        const auto earlier = rig.shape.backbone(10.05, snapshot, nodes, 0, false);
        const auto later = rig.shape.backbone(10.1, snapshot, nodes, 0, false);

        ASSERT_TRUE(earlier && later);
        EXPECT_DOUBLE_EQ(earlier->points.front().pose(2, 3), rig.nodePose(0, 0, snapshot_values).z());
        EXPECT_DOUBLE_EQ(later->points.front().pose(2, 3), rig.nodePose(1, 0, snapshot_values).z());
    }

    TEST(RobotShape, BackboneIsNulloptWithoutARowInTheSnapshot)
    {
        ShapeRig rig;
        rig.shape.bindPriorInterpolator(rig.prior);
        rig.addRow(10.0);
        const auto nodes = rig.state->snapshotEstimationNodes();

        EXPECT_FALSE(rig.shape.backbone(9.0, ShapeRig::snapshot(10.0, rig.values), nodes, 0, false));           // before the first row
        EXPECT_FALSE(rig.shape.backbone(10.0, ShapeRig::snapshot(9.9, gtsam::Values()), nodes, 0, false));      // snapshot older than every row
        EXPECT_FALSE(rig.shape.backbone(10.0, ShapeRig::snapshot(10.0, gtsam::Values()), nodes, 0, false));     // row marginalized out
        EXPECT_FALSE(rig.shape.backbone(10.0, ShapeRig::snapshot(10.0, rig.values), {}, 0, false));             // no nodes yet
    }

    TEST(RobotShape, BackboneWithoutInterpolatorHoldsOnlyTheNodes)
    {
        ShapeRig rig;
        rig.addRow(0.0);

        const auto backbone = rig.shape.backbone(0.0, ShapeRig::snapshot(0.0, rig.values), rig.state->snapshotEstimationNodes(), 3, true);

        ASSERT_TRUE(backbone);
        EXPECT_EQ(backbone->points_per_segment, 1u);
        ASSERT_EQ(backbone->points.size(), kSpatialDim);
        for (unsigned int i = 0; i < kSpatialDim; i++)
            EXPECT_TRUE(Pose(backbone->points[i].pose).equals(rig.nodePose(0, i, rig.values), 1e-12)) << "node " << i;
    }

    TEST(RobotShape, TemporalInterpolationMovesBetweenTheBorderingRows)
    {
        ShapeRig rig;
        rig.shape.bindPriorInterpolator(rig.prior);
        rig.addRow(10.0);
        rig.addRow(10.1);
        gtsam::Values snapshot_values = rig.values;
        rig.offsetRow(1, snapshot_values, 0.5);
        const auto snapshot = ShapeRig::snapshot(10.1, snapshot_values);
        const auto nodes = rig.state->snapshotEstimationNodes();
        const double z0 = rig.nodePose(0, 0, snapshot_values).z(), z1 = rig.nodePose(1, 0, snapshot_values).z();

        const auto snapped = rig.shape.backbone(10.05, snapshot, nodes, 0, false);
        const auto interpolated = rig.shape.backbone(10.05, snapshot, nodes, 0, true);
        const auto pose = rig.shape.poseAt(10.05, 0.0, snapshot, nodes, true);

        ASSERT_TRUE(snapped && interpolated && pose);
        EXPECT_DOUBLE_EQ(snapped->points.front().pose(2, 3), z0);
        EXPECT_GT(interpolated->points.front().pose(2, 3), z0 + 1e-6);
        EXPECT_LT(interpolated->points.front().pose(2, 3), z1 - 1e-6);
        EXPECT_NEAR((*pose)(2, 3), interpolated->points.front().pose(2, 3), 1e-12);
    }
} // namespace Spacetime
