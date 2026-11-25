#include <cmath>
#include <memory>
#include <optional>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>
#include <vtkActor.h>
#include <vtkMatrix4x4.h>

#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/joint_models/EndEffector.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/visualizer/default/objects/ContinuumRobot.hpp>
#include <spacetime/visualizer/default/objects/EndEffector.hpp>
#include <spacetime/visualizer/default/objects/Joint.hpp>

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using Vector6 = Eigen::Matrix<double, 6, 1>;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;
        using EndEffector = Joints::EndEffector<Pose>;
        using EEValues = Joints::EndEffectorNodeValues<Pose>;
        using EEView = Visualization::Default::EndEffector<Pose>;
        using RobotNode = Priors::WNOA::Node<Pose>;
        using RobotState = Types::ContinuumRobotState<RobotNode>;
        using RobotPrior = Priors::WNOA::Prior<Pose>;
        using RobotView = Visualization::Default::ContinuumRobot<RobotNode>;

        constexpr double kPeriod = 0.1;
        constexpr double kLength = 0.3;

        Eigen::Matrix4d toEigen(vtkMatrix4x4 *matrix)
        {
            Eigen::Matrix4d out;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    out(r, c) = matrix->GetElement(r, c);
            return out;
        }

        /// An upright robot whose tip is attached to an end effector 5 cm beyond it, both
        /// extended over rows at t = 0 and kPeriod, the end effector moving between them.
        struct Rig
        {
            std::shared_ptr<RobotState> state = std::make_shared<RobotState>(kLength, 4);
            std::shared_ptr<RobotPrior> robot_prior;
            Pose ee0 = Pose(gtsam::Rot3::Ry(-M_PI / 2), gtsam::Point3(0, 0, kLength + 0.05));
            std::shared_ptr<EndEffector> ee = std::make_shared<EndEffector>(ee0);
            std::shared_ptr<Joints::Joint<RobotPrior, Joints::EndEffectorPrior<Pose>>> joint;
            gtsam::Values values;

            Rig()
            {
                Priors::WNOA::NodeValues<Pose> reference;
                reference.pose(Pose(gtsam::Rot3::Ry(-M_PI / 2), gtsam::Point3(0, 0, 0)));
                reference.epsilon((Vector6() << 0, 0, 0, 1, 0, 0).finished());
                robot_prior = std::make_shared<RobotPrior>(state, reference);
                Joints::JointOptions options;
                options.offset = Pose(gtsam::Rot3(), gtsam::Point3(0.05, 0, 0));
                joint = ee->attach(robot_prior, kLength, Matrix6::Identity() * 1e-6, options);

                KeyManager km;
                for (double t : {0.0, kPeriod})
                    for (const std::shared_ptr<Types::Model> &model : std::vector<std::shared_ptr<Types::Model>>{robot_prior, ee})
                    {
                        gtsam::Values new_values;
                        model->extendState(km, t, values, new_values);
                        values.insert(new_values);
                    }
                const auto second = ee->state()->estimation_nodes[1];
                values.update(second->poseKey(), ee0 * Pose(gtsam::Rot3::Rz(0.2), gtsam::Point3(0.0, 0.03, 0.0)));
            }

            Pose estimated(double t) const
            {
                return std::dynamic_pointer_cast<EEValues>(ee->endEffectorPrior()->query(t, 0.0, values))->pose();
            }

            std::shared_ptr<EEView> view() const
            {
                auto view = std::make_shared<EEView>(ee);
                auto snapshot = std::make_shared<const EstimatorSnapshot>(EstimatorSnapshot{kPeriod, values});
                view->bindSnapshotSource([snapshot](double) { return snapshot; });
                return view;
            }
        };
    } // namespace

    TEST(EndEffectorVisualization, FrameIsDrawnAtTheEstimatedPose)
    {
        Rig rig;
        auto view = rig.view();
        view->initActors();
        view->toggleTemporalInterpolation(); // between rows too (see the next test for the default)
        EXPECT_TRUE(view->temporalInterpolationEnabled());
        for (double t : {0.0, 0.5 * kPeriod, kPeriod})
        {
            view->updateVisualization(t, kPeriod);
            ASSERT_TRUE(view->mp_axes);
            EXPECT_TRUE(view->mp_axes->GetVisibility()) << t;
            EXPECT_TRUE(toEigen(view->mp_axes->GetUserMatrix()).isApprox(rig.estimated(t).matrix(), 1e-9)) << t;
            const auto pose = view->poseAt(t, kPeriod);
            ASSERT_TRUE(pose.has_value());
            EXPECT_TRUE(pose->isApprox(rig.estimated(t).matrix(), 1e-9));
            EXPECT_TRUE(view->pointPose()(t, kPeriod)->isApprox(*pose, 1e-12));
        }
    }

    // Like a robot view, the frame snaps to the latest row at or before the display time unless
    // temporal interpolation is on -- and so does its pointPose(), which places its joints.
    TEST(EndEffectorVisualization, FrameSnapsToTheLatestRowUnlessTemporalInterpolationIsOn)
    {
        Rig rig;
        auto view = rig.view();
        view->initActors();
        EXPECT_FALSE(view->temporalInterpolationEnabled());
        const auto point_pose = view->pointPose();
        const double t = 0.5 * kPeriod;
        view->updateVisualization(t, kPeriod);
        EXPECT_TRUE(toEigen(view->mp_axes->GetUserMatrix()).isApprox(rig.estimated(0.0).matrix(), 1e-9));
        EXPECT_TRUE(view->poseAt(t, kPeriod)->isApprox(rig.estimated(0.0).matrix(), 1e-9));
        EXPECT_TRUE(point_pose(t, kPeriod)->isApprox(rig.estimated(0.0).matrix(), 1e-9));

        view->toggleTemporalInterpolation();
        EXPECT_FALSE(rig.estimated(t).matrix().isApprox(rig.estimated(0.0).matrix(), 1e-6)); // the rows differ
        view->updateVisualization(t, kPeriod);
        EXPECT_TRUE(toEigen(view->mp_axes->GetUserMatrix()).isApprox(rig.estimated(t).matrix(), 1e-9));
        EXPECT_TRUE(point_pose(t, kPeriod)->isApprox(rig.estimated(t).matrix(), 1e-9));
    }

    TEST(EndEffectorVisualization, HiddenWithoutASnapshot)
    {
        Rig rig;
        auto view = std::make_shared<EEView>(rig.ee);
        view->initActors();
        view->updateVisualization(0.0, 0.0);
        EXPECT_FALSE(view->mp_axes->GetVisibility());
        EXPECT_FALSE(view->poseAt(0.0, 0.0).has_value());
    }

    TEST(EndEffectorVisualization, ToggledOffFrameStaysHidden)
    {
        Rig rig;
        auto view = rig.view();
        view->initActors();
        view->toggleVisibility();
        view->updateVisualization(0.0, kPeriod);
        EXPECT_FALSE(view->mp_axes->GetVisibility());
    }

    TEST(EndEffectorVisualization, JointLinksTheRobotPointToTheFrame)
    {
        Rig rig;
        auto robot_view = std::make_shared<RobotView>(rig.state);
        robot_view->bindPriorInterpolator(rig.robot_prior);
        auto snapshot = std::make_shared<const EstimatorSnapshot>(EstimatorSnapshot{kPeriod, rig.values});
        robot_view->bindSnapshotSource([snapshot](double) { return snapshot; });
        auto ee_view = rig.view();

        auto joint_view = Visualization::Default::Joint::of(*rig.joint, robot_view, ee_view);
        ASSERT_TRUE(joint_view->drawsLink());
        joint_view->initActors();
        joint_view->updateVisualization(kPeriod, kPeriod);
        const Eigen::Matrix4d T = toEigen(joint_view->mp_actor->GetUserMatrix());
        const Eigen::Vector3d tip = robot_view->poseAt(kPeriod, kLength, kPeriod)->block<3, 1>(0, 3);
        const Eigen::Vector3d frame = rig.estimated(kPeriod).translation();
        EXPECT_TRUE(((T * Eigen::Vector4d(0, -0.5, 0, 1)).head<3>().isApprox(tip, 1e-9)));
        EXPECT_TRUE(((T * Eigen::Vector4d(0, 0.5, 0, 1)).head<3>().isApprox(frame, 1e-9)));
    }

    TEST(EndEffectorVisualization, FrameMatrixIsUpdatedInPlace)
    {
        Rig rig;
        auto view = rig.view();
        view->initActors();
        view->toggleTemporalInterpolation();
        view->updateVisualization(0.0, kPeriod);
        vtkMatrix4x4 *matrix = view->mp_axes->GetUserMatrix();
        ASSERT_NE(matrix, nullptr);

        view->updateVisualization(kPeriod, kPeriod);
        EXPECT_EQ(view->mp_axes->GetUserMatrix(), matrix);
        EXPECT_TRUE(toEigen(matrix).isApprox(rig.estimated(kPeriod).matrix(), 1e-9));
    }

    TEST(EndEffectorVisualization, FrameDrawsAGivenMapper)
    {
        Rig rig;
        const auto mapper = Visualization::Default::Utilities::createFrameMapper(0.04, 0.06);
        auto a = rig.view(), b = rig.view();
        a->initActors(mapper);
        b->initActors(mapper);
        EXPECT_EQ(a->mp_axes->GetMapper(), mapper.Get());
        EXPECT_EQ(b->mp_axes->GetMapper(), mapper.Get());

        auto own = rig.view();
        own->initActors();
        ASSERT_NE(own->mp_axes->GetMapper(), nullptr);
        EXPECT_NE(own->mp_axes->GetMapper(), mapper.Get());
    }
} // namespace Spacetime
