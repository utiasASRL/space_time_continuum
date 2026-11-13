#include <memory>

#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>
#include <vtkActor.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/visualizer/default/objects/ContinuumRobot.hpp>

namespace Spacetime::Priors::WNOV
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using PriorType = Prior<Pose>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;

        constexpr unsigned int kSpatialDim = 3;
        constexpr double kDrawTol = 1e-6; // vtkPoints stores floats

        /// A WNOV robot with its view's backbone/frame actors set up directly (as
        /// Default::ContinuumRobot::visualize() would), bound to the prior's own interpolator as
        /// the demo does, so updateVisualization() runs without a Qt/VTK window.
        struct Rig
        {
            std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, kSpatialDim);
            std::shared_ptr<Visualization::Default::ContinuumRobot<NodeType>> view = std::make_shared<Visualization::Default::ContinuumRobot<NodeType>>(state);
            std::shared_ptr<PriorType> prior = std::make_shared<PriorType>(state, Pose(), (Vector6() << 0, 0, 0, 0, 0, 1.0).finished());
            gtsam::Values values;
            KeyManager km;

            Rig()
            {
                view->bindInterpolator([p = prior](double time, double arclength, const gtsam::Values &v,
                                                    const std::shared_ptr<NodeType> &node00, const std::shared_ptr<NodeType> &node01,
                                                    const std::shared_ptr<NodeType> &node10, const std::shared_ptr<NodeType> &node11,
                                                    const PosteriorCovariance *posterior, Eigen::MatrixXd *posteriorCovariance)
                                        { return p->interpolator().interpolate(time, arclength, v, node00, node01, node10, node11,
                                                                               nullptr, nullptr, posterior, posteriorCovariance); });
                view->rebuildBackboneGeometry();
                view->mp_backbone_actor = vtkSmartPointer<vtkActor>::New();
                for (unsigned int i = 0; i < kSpatialDim; i++)
                    view->mp_axes.push_back(vtkSmartPointer<vtkActor>::New());
            }

            void addRow(double time)
            {
                gtsam::Values new_values;
                prior->extendState(km, time, values, new_values);
                values.insert(new_values);
            }

            void publish(double time)
            {
                auto snapshot = std::make_shared<const EstimatorSnapshot>(EstimatorSnapshot{time, values});
                view->bindSnapshotSource([snapshot](double)
                                          { return snapshot; });
            }

            gtsam::Point3 drawnPoint(unsigned int index) const
            {
                double point[3];
                view->mp_backbone_points->GetPoint(index, point);
                return gtsam::Point3(point[0], point[1], point[2]);
            }

            gtsam::Point3 nodePosition(unsigned int row, unsigned int n) const
            {
                return values.at<Pose>(state->estimation_nodes.at(row * kSpatialDim + n)->poseKey()).translation();
            }
        };
    } // namespace

    TEST(WNOVVisualization, BackboneIsPiecewiseLinearBetweenNodes)
    {
        Rig rig;
        rig.addRow(0.0);
        // Bend the robot: move the middle node off the straight line (translation only,
        // so the geodesic between nodes is a straight segment).
        const auto &middle = rig.state->estimation_nodes.at(1);
        rig.values.update(middle->poseKey(), Pose(gtsam::Rot3(), gtsam::Point3(0.2, 0.0, 0.5)));
        rig.publish(0.0);

        rig.view->adjustSpatialInterpolationPoints(+3); // 3 extra points per segment
        rig.view->updateVisualization(0.0, 0.0);
        ASSERT_TRUE(rig.view->mp_backbone_actor->GetVisibility());
        ASSERT_EQ(rig.view->mp_backbone_points->GetNumberOfPoints(), 3 + 2 * 3);

        for (unsigned int segment = 0; segment < 2; segment++)
        {
            const gtsam::Point3 a = rig.nodePosition(0, segment), b = rig.nodePosition(0, segment + 1);
            EXPECT_TRUE(gtsam::assert_equal(a, rig.drawnPoint(segment * 4), kDrawTol));
            for (unsigned int k = 1; k < 4; k++)
                EXPECT_TRUE(gtsam::assert_equal(gtsam::Point3(a + (k / 4.0) * (b - a)), rig.drawnPoint(segment * 4 + k), kDrawTol))
                    << "segment " << segment << ", point " << k;
        }
    }

    TEST(WNOVVisualization, TemporalInterpolationIsLinearBetweenRows)
    {
        Rig rig;
        rig.addRow(0.0);
        rig.addRow(0.1);
        // Shift the whole second row, so the drawn robot at t = 0.025 is a quarter of the way.
        for (unsigned int n = 0; n < kSpatialDim; n++)
        {
            const gtsam::Key key = rig.state->estimation_nodes.at(kSpatialDim + n)->poseKey();
            rig.values.update(key, Pose(gtsam::Rot3(), rig.values.at<Pose>(key).translation() + gtsam::Point3(0.4, 0.0, 0.0)));
        }
        rig.publish(0.1);

        rig.view->toggleTemporalInterpolation();
        rig.view->updateVisualization(0.025, 0.1);
        ASSERT_TRUE(rig.view->mp_backbone_actor->GetVisibility());
        for (unsigned int n = 0; n < kSpatialDim; n++)
        {
            const gtsam::Point3 expected = 0.75 * rig.nodePosition(0, n) + 0.25 * rig.nodePosition(1, n);
            EXPECT_TRUE(gtsam::assert_equal(expected, rig.drawnPoint(n), kDrawTol)) << "node " << n;
        }
    }
} // namespace Spacetime::Priors::WNOV
