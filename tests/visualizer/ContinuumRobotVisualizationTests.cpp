#include <memory>
#include <utility>
#include <vector>

#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>
#include <vtkActor.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkMatrix4x4.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/visualizer/default/objects/ContinuumRobot.hpp>
#include "ContinuumRobotRig.hpp"

namespace Spacetime
{
    using namespace VisualizerTests;

    // Online, estimate(t) appends its row before solving but only publishes the
    // snapshot holding it afterwards. A frame drawn mid-solve must fall back to the
    // newest row the snapshot actually holds instead of hiding the robot.
    TEST(ContinuumRobotVisualization, MidSolveRowNotYetInSnapshotFallsBackToLatestSnapshotRow)
    {
        Rig rig;
        rig.addRow(10.0);
        const gtsam::Values snapshot_values = rig.values; // row 0 only
        rig.publish(10.0, snapshot_values);
        rig.addRow(10.1); // estimate(10.1) has started: row 1 exists, but the snapshot is still 10.0

        rig.view->updateVisualization(10.15, 10.15);

        EXPECT_TRUE(rig.backboneVisible());
        EXPECT_DOUBLE_EQ(rig.drawnBaseZ(), rig.rowBaseZ(0, snapshot_values));
    }

    // Offline (and online with a phase delay), the display time can be earlier than
    // the snapshot time. The cap must not change which row is drawn then.
    TEST(ContinuumRobotVisualization, DisplayTimeBeforeSnapshotTimeStillSelectsByDisplayTime)
    {
        Rig rig;
        rig.addRow(10.0);
        rig.addRow(10.1);
        gtsam::Values snapshot_values = rig.values;
        rig.offsetRow(1, snapshot_values, 0.5);
        rig.publish(10.1, snapshot_values);

        rig.view->updateVisualization(10.05, 10.1);

        EXPECT_TRUE(rig.backboneVisible());
        EXPECT_DOUBLE_EQ(rig.drawnBaseZ(), rig.rowBaseZ(0, snapshot_values));
    }

    // Batch mode publishes one snapshot of the whole solution, at the last row's time, and
    // the demo serves it at every playback time: each row must be drawn at its own time.
    TEST(ContinuumRobotVisualization, BatchSnapshotDrawsEveryRowAtItsDisplayTime)
    {
        Rig rig;
        for (int row = 0; row < 4; row++)
            rig.addRow(10.0 + row * kPeriod);
        gtsam::Values snapshot_values = rig.values;
        for (unsigned int row = 0; row < 4; row++)
            rig.offsetRow(row, snapshot_values, 0.5 * row);
        rig.publish(10.0 + 3 * kPeriod, snapshot_values);

        for (unsigned int row = 0; row < 4; row++)
        {
            const double time = 10.0 + row * kPeriod;
            rig.view->updateVisualization(time, time);
            ASSERT_TRUE(rig.backboneVisible()) << "row " << row;
            EXPECT_DOUBLE_EQ(rig.drawnBaseZ(), rig.rowBaseZ(row, snapshot_values)) << "row " << row;
        }
    }

    TEST(ContinuumRobotVisualization, DisplayTimeBeforeFirstRowHidesRobot)
    {
        Rig rig;
        rig.addRow(10.0);
        rig.publish(10.0, rig.values);

        rig.view->updateVisualization(9.0, 9.0);

        EXPECT_FALSE(rig.backboneVisible());
    }

    TEST(ContinuumRobotVisualization, SnapshotOlderThanEveryRowHidesRobot)
    {
        Rig rig;
        rig.addRow(10.0);
        rig.publish(9.9, gtsam::Values()); // e.g. published before any row existed

        rig.view->updateVisualization(10.05, 10.05);

        EXPECT_FALSE(rig.backboneVisible());
    }

    TEST(ContinuumRobotVisualization, NoSnapshotHidesRobot)
    {
        Rig rig;
        rig.addRow(10.0);
        rig.view->bindSnapshotSource([](double)
                                      { return std::shared_ptr<const EstimatorSnapshot>(); });

        rig.view->updateVisualization(10.05, 10.05);

        EXPECT_FALSE(rig.backboneVisible());
    }

    TEST(ContinuumRobotVisualization, PoseAtMatchesNodesAndInterpolatesBetweenThem)
    {
        Rig rig;
        rig.bindInterpolator();
        rig.addRow(0.0);
        rig.addRow(kPeriod);
        EXPECT_FALSE(rig.view->poseAt(0.0, 1.0 / 3.0, kPeriod).has_value()); // nothing published yet
        rig.publish(kPeriod, rig.values);

        const auto node = rig.state->getClosestShiftedNode(1.0 / 3.0, 0.0, -1, -1);
        const auto on_node = rig.view->poseAt(0.0, node->arclength, kPeriod);
        ASSERT_TRUE(on_node.has_value());
        EXPECT_TRUE(on_node->isApprox(rig.values.at<Pose>(node->poseKey()).matrix(), 1e-9));

        // Straight along z: between the s = 1/3 and s = 2/3 nodes, z is the arclength.
        const auto between = rig.view->poseAt(0.5 * kPeriod, 0.5, kPeriod);
        ASSERT_TRUE(between.has_value());
        EXPECT_NEAR((*between)(2, 3), 0.5, 1e-6);

        EXPECT_FALSE(rig.view->poseAt(-1.0, 0.5, kPeriod).has_value()); // before the first row
        EXPECT_FALSE(rig.view->poseAt(0.0, -0.5, kPeriod).has_value()); // before the base
    }

    // poseAt() resolves time exactly as the backbone is drawn: snapped to the latest row at or
    // before the display time unless temporal interpolation is on, so anything placed with it
    // (joints, end-effector frames) moves in step with the drawn robot.
    TEST(ContinuumRobotVisualization, PoseAtSnapsToTheLatestRowUnlessTemporalInterpolationIsOn)
    {
        Rig rig;
        rig.bindInterpolator();
        rig.addRow(0.0);
        rig.addRow(kPeriod);
        gtsam::Values shifted = rig.values;
        rig.offsetRow(1, shifted, 0.2); // row 1 is 0.2 higher than row 0
        rig.publish(kPeriod, shifted);
        const double row0_z = rig.rowBaseZ(0, shifted), row1_z = rig.rowBaseZ(1, shifted);

        EXPECT_FALSE(rig.view->temporalInterpolationEnabled());
        const auto snapped = rig.view->poseAt(0.5 * kPeriod, 0.0, kPeriod);
        ASSERT_TRUE(snapped.has_value());
        EXPECT_NEAR((*snapped)(2, 3), row0_z, 1e-9);
        rig.view->updateVisualization(0.5 * kPeriod, kPeriod);
        EXPECT_NEAR(rig.drawnBaseZ(), (*snapped)(2, 3), 1e-6); // where the backbone is drawn (VTK points are floats)

        rig.view->toggleTemporalInterpolation();
        EXPECT_TRUE(rig.view->temporalInterpolationEnabled());
        const auto interpolated = rig.view->poseAt(0.5 * kPeriod, 0.0, kPeriod);
        ASSERT_TRUE(interpolated.has_value());
        EXPECT_GT((*interpolated)(2, 3), row0_z + 0.05);
        EXPECT_LT((*interpolated)(2, 3), row1_z - 0.05);
        rig.view->updateVisualization(0.5 * kPeriod, kPeriod);
        EXPECT_NEAR(rig.drawnBaseZ(), (*interpolated)(2, 3), 1e-6);

        // On a row, both agree.
        EXPECT_NEAR((*rig.view->poseAt(kPeriod, 0.0, kPeriod))(2, 3), row1_z, 1e-9);
        rig.view->toggleTemporalInterpolation();
        EXPECT_NEAR((*rig.view->poseAt(kPeriod, 0.0, kPeriod))(2, 3), row1_z, 1e-9);
    }

    TEST(ContinuumRobotVisualization, PoseAtIsUnavailableOnceMarginalized)
    {
        Rig rig;
        rig.bindInterpolator();
        rig.addRow(0.0);
        rig.addRow(kPeriod);
        gtsam::Values window = rig.values;
        for (unsigned int i = 0; i < kSpatialDim; i++)
            for (const gtsam::Key key : rig.state->estimation_nodes.at(i)->keys())
                window.erase(key); // row 0 left the window
        rig.publish(kPeriod, window);
        EXPECT_FALSE(rig.view->poseAt(0.5 * kPeriod, 0.5, kPeriod).has_value());
        EXPECT_TRUE(rig.view->poseAt(kPeriod, 0.5, kPeriod).has_value());
    }

    TEST(ContinuumRobotVisualization, BindPriorInterpolatorUsesThePriorsInterpolator)
    {
        Rig manual, bound;
        manual.bindInterpolator();
        bound.view->bindPriorInterpolator(bound.prior);
        for (Rig *rig : {&manual, &bound})
        {
            rig->addRow(0.0);
            rig->addRow(kPeriod);
            rig->publish(kPeriod, rig->values);
        }
        for (const auto &[time, arclength] : std::vector<std::pair<double, double>>{{0.0, 0.0}, {0.5 * kPeriod, 0.5}, {kPeriod, 0.9}})
        {
            const auto expected = manual.view->poseAt(time, arclength, kPeriod), actual = bound.view->poseAt(time, arclength, kPeriod);
            ASSERT_TRUE(expected.has_value() && actual.has_value());
            EXPECT_TRUE(actual->isApprox(*expected, 1e-12)) << time << ", " << arclength;
        }
    }

    TEST(ContinuumRobotVisualization, BindPriorInterpolatorRequiresThePriorOfTheDrawnState)
    {
        Rig rig, other;
        EXPECT_THROW(rig.view->bindPriorInterpolator(std::shared_ptr<PriorT>()), std::invalid_argument);
        EXPECT_THROW(rig.view->bindPriorInterpolator(other.prior), std::invalid_argument); // extends another state
        rig.addRow(0.0);
        rig.publish(0.0, rig.values);
        EXPECT_FALSE(rig.view->poseAt(0.0, 0.0, 0.0).has_value()); // the rejected priors bound nothing
        EXPECT_NO_THROW(rig.view->bindPriorInterpolator(rig.prior));
        EXPECT_TRUE(rig.view->poseAt(0.0, 0.0, 0.0).has_value());
    }

    TEST(ContinuumRobotVisualization, NodeFrameMatricesAreUpdatedInPlace)
    {
        Rig rig;
        rig.addRow(0.0);
        rig.publish(0.0, rig.values);
        rig.view->updateVisualization(0.0, 0.0);
        vtkMatrix4x4 *matrix = rig.view->mp_axes[0]->GetUserMatrix();
        ASSERT_NE(matrix, nullptr);

        gtsam::Values moved = rig.values;
        rig.offsetRow(0, moved, 0.5);
        rig.publish(0.0, moved);
        rig.view->updateVisualization(0.0, 0.0);
        EXPECT_EQ(rig.view->mp_axes[0]->GetUserMatrix(), matrix);
        EXPECT_DOUBLE_EQ(matrix->GetElement(2, 3), rig.rowBaseZ(0, moved));
    }

    TEST(ContinuumRobotVisualization, NodeFramesToggleIndependentlyOfTheBackbone)
    {
        Rig rig;
        rig.drawOneRow();
        rig.view->toggleNodeFrames();
        EXPECT_FALSE(rig.view->nodeFramesShown());
        EXPECT_FALSE(rig.nodeFramesVisible()) << "hidden immediately";
        EXPECT_TRUE(rig.backboneVisible());
        rig.view->updateVisualization(0.0, 0.0);
        EXPECT_FALSE(rig.nodeFramesVisible()) << "and kept hidden by later frames";
        EXPECT_TRUE(rig.backboneVisible());

        rig.view->toggleNodeFrames();
        EXPECT_TRUE(rig.view->nodeFramesShown());
        EXPECT_TRUE(rig.nodeFramesVisible());
        rig.view->updateVisualization(0.0, 0.0);
        EXPECT_TRUE(rig.nodeFramesVisible());
    }

    TEST(ContinuumRobotVisualization, NodeFramesCanStartHidden)
    {
        Rig rig;
        rig.view->show_node_frames = false;
        EXPECT_FALSE(rig.view->nodeFramesShown());
        rig.drawOneRow();
        EXPECT_TRUE(rig.backboneVisible());
        EXPECT_FALSE(rig.nodeFramesVisible());
    }

    TEST(ContinuumRobotVisualization, ShowingTheRobotAgainKeepsHiddenNodeFramesHidden)
    {
        Rig rig;
        rig.drawOneRow();
        rig.view->toggleNodeFrames();
        rig.view->toggleVisibility(); // B off
        EXPECT_FALSE(rig.backboneVisible());
        EXPECT_FALSE(rig.nodeFramesVisible());
        rig.view->toggleVisibility(); // B on
        EXPECT_TRUE(rig.backboneVisible());
        EXPECT_FALSE(rig.nodeFramesVisible());
        rig.view->updateVisualization(0.0, 0.0);
        EXPECT_TRUE(rig.backboneVisible());
        EXPECT_FALSE(rig.nodeFramesVisible());
    }

    TEST(ContinuumRobotVisualization, BackboneSegmentsAlternateTheLightShades)
    {
        Rig rig;
        for (int extra_points : {0, 2})
        {
            rig.view->m_spatial_interp_points = static_cast<unsigned int>(extra_points);
            rig.view->rebuildBackboneGeometry();
            vtkDataArray *colors = rig.view->mp_backbone_polydata->GetCellData()->GetScalars();
            ASSERT_NE(colors, nullptr);
            const int cells_per_segment = extra_points + 1;
            ASSERT_EQ(colors->GetNumberOfTuples(), (kSpatialDim - 1) * cells_per_segment);
            for (vtkIdType cell = 0; cell < colors->GetNumberOfTuples(); cell++)
            {
                const unsigned char expected = Visualization::Default::Utilities::kBackboneShades[(cell / cells_per_segment) % 2];
                double rgb[3];
                colors->GetTuple(cell, rgb);
                for (double channel : rgb)
                    EXPECT_DOUBLE_EQ(channel, expected) << "cell " << cell << " with " << extra_points << " extra points";
            }
        }
    }
} // namespace Spacetime
