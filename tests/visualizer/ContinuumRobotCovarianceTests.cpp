#include <memory>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkRenderer.h>
#include <vtkTubeFilter.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/visualizer/default/utilities.hpp>
#include <spacetime/visualizer/default/objects/ContinuumRobot.hpp>

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using ValuesType = Priors::WNOA::Values<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ViewType = Visualization::Default::ContinuumRobot<NodeType>;
        using PriorType = Priors::WNOA::Prior<Pose>;
        using InterpolatorType = Priors::WNOA::Interpolator<Pose>;
        using MatrixN = Eigen::Matrix<double, 6, 6>;

        constexpr unsigned int kSpatialDim = 3;

        /// A robot, its prior and an estimator solved over two rows (t = 0, 0.1), with
        /// the robot view's VTK objects created directly instead of through a Visualizer
        /// (which needs a live OpenGL window), so updateVisualization can run headless.
        struct RobotFixture
        {
            std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, kSpatialDim);
            std::shared_ptr<ViewType> view = std::make_shared<ViewType>(state);
            std::shared_ptr<PriorType> prior;
            std::unique_ptr<Estimator> estimator;

            /// @param prefill_ellipsoids Create the ellipsoid actors up front; otherwise the view creates them on the first toggleCovariance().
            explicit RobotFixture(bool compute_covariance, bool prefill_ellipsoids = true)
            {
                ValuesType node_reference;
                node_reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
                prior = std::make_shared<PriorType>(state, node_reference);

                EstimatorOptions options;
                options.smootherLag = 1.0;
                options.compute_covariance = compute_covariance;
                estimator = std::make_unique<Estimator>(options);
                estimator->addPrior(prior);
                estimator->estimate(0.0);
                estimator->estimate(0.1);

                auto snapshot = estimator->getSnapshot();
                view->bindSnapshotSource([snapshot](double)
                                          { return snapshot; });
                auto interpolator = std::make_shared<InterpolatorType>(MatrixN::Identity(), MatrixN::Identity(), MatrixN::Identity(), MatrixN::Identity());
                view->bindInterpolator([interpolator](double time, double arclength, const gtsam::Values &values,
                                                       const std::shared_ptr<NodeType> &node00, const std::shared_ptr<NodeType> &node01,
                                                       const std::shared_ptr<NodeType> &node10, const std::shared_ptr<NodeType> &node11,
                                                       const PosteriorCovariance *posterior, Eigen::MatrixXd *posteriorCovariance)
                                        { return interpolator->interpolate(time, arclength, values, node00, node01, node10, node11,
                                                                           nullptr, nullptr, posterior, posteriorCovariance); });

                view->mp_backbone_tube = vtkSmartPointer<vtkTubeFilter>::New();
                view->rebuildBackboneGeometry();
                view->mp_backbone_actor = vtkSmartPointer<vtkActor>::New();
                for (unsigned int i = 0; i < kSpatialDim; i++)
                    view->mp_axes.push_back(Visualization::Default::Utilities::createFrameActor(0.01, 0.01));
                view->mp_covariance_mapper = Visualization::Default::Utilities::createUnitSphereMapper();
                for (unsigned int i = 0; prefill_ellipsoids && i < kMaxEllipsoids; i++)
                    view->mp_covariance_actors.push_back(Visualization::Default::Utilities::createEllipsoidActor(view->mp_covariance_mapper));
            }

            static constexpr unsigned int kMaxEllipsoids = kSpatialDim + (kSpatialDim - 1) * ViewType::kMaxSpatialInterpolationPoints;

            unsigned int visibleEllipsoids() const
            {
                unsigned int count = 0;
                for (const auto &actor : view->mp_covariance_actors)
                    count += actor->GetVisibility() ? 1 : 0;
                return count;
            }
        };

        Eigen::Matrix4d toEigen(vtkMatrix4x4 *matrix)
        {
            Eigen::Matrix4d out;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    out(r, c) = matrix->GetElement(r, c);
            return out;
        }

        /// Check that `actor` draws the 99.7% position ellipsoid of a node whose pose is
        /// `pose` and whose pose-tangent posterior covariance leads `covariance`.
        void expectEllipsoidMatches(vtkActor *actor, const Pose &pose, const Eigen::MatrixXd &covariance)
        {
            ASSERT_TRUE(actor->GetVisibility());
            const Eigen::Matrix4d transform = toEigen(actor->GetUserMatrix());
            const Eigen::Matrix3d R = pose.rotation().matrix();
            const Eigen::Matrix3d expected = R * covariance.block(3, 3, 3, 3) * R.transpose();
            const double s = Visualization::Default::Utilities::kEllipsoidScale99_7;
            const Eigen::Matrix3d L = transform.topLeftCorner(3, 3);
            EXPECT_TRUE((L * L.transpose()).isApprox(s * s * expected, 1e-8));
            EXPECT_TRUE(transform.col(3).head(3).isApprox(pose.translation(), 1e-12));
        }
    } // namespace

    TEST(ContinuumRobotCovariance, EllipsoidsAreHiddenUntilToggled)
    {
        RobotFixture f(/*compute_covariance=*/true);
        f.view->updateVisualization(0.1, 0.1);
        EXPECT_EQ(f.visibleEllipsoids(), 0u);
        EXPECT_TRUE(f.view->mp_backbone_actor->GetVisibility()) << "the mean is still drawn";
    }

    TEST(ContinuumRobotCovariance, OneEllipsoidPerNodeFromTheNodeMarginal)
    {
        RobotFixture f(/*compute_covariance=*/true);
        f.view->toggleCovariance();
        f.view->updateVisualization(0.1, 0.1);
        ASSERT_EQ(f.visibleEllipsoids(), kSpatialDim);

        const auto posterior = f.estimator->getCurrentCovariance();
        const gtsam::Values values = f.estimator->getCurrentValues();
        for (unsigned int i = 0; i < kSpatialDim; i++)
        {
            SCOPED_TRACE(i);
            const auto &node = f.state->estimation_nodes[kSpatialDim + i]; // the t = 0.1 row
            expectEllipsoidMatches(f.view->mp_covariance_actors[i], node->getValues(values).pose(), posterior->covariance(node->poseKey()));
        }

        f.view->toggleCovariance();
        f.view->updateVisualization(0.1, 0.1);
        EXPECT_EQ(f.visibleEllipsoids(), 0u);
    }

    TEST(ContinuumRobotCovariance, InterpolatedPointsGetInterpolatedCovariances)
    {
        RobotFixture f(/*compute_covariance=*/true);
        f.view->toggleCovariance();
        f.view->toggleTemporalInterpolation();
        f.view->adjustSpatialInterpolationPoints(+2);
        const double time = 0.05; // between the two rows
        f.view->updateVisualization(time, 0.1);

        const unsigned int points_per_segment = 3;
        ASSERT_EQ(f.visibleEllipsoids(), kSpatialDim + (kSpatialDim - 1) * 2);

        const auto posterior = f.estimator->getCurrentCovariance();
        const gtsam::Values values = f.estimator->getCurrentValues();
        const InterpolatorType interpolator(MatrixN::Identity(), MatrixN::Identity(), MatrixN::Identity(), MatrixN::Identity());
        const auto &n = f.state->estimation_nodes; // row-major: n[row * 3 + s]

        // A true node, temporally interpolated between its two rows.
        Eigen::MatrixXd covariance;
        auto mean = std::static_pointer_cast<ValuesType>(
            interpolator.interpolate(time, n[1]->arclength, values, n[1], n[4], nullptr, nullptr, nullptr, nullptr, posterior.get(), &covariance));
        expectEllipsoidMatches(f.view->mp_covariance_actors[1 * points_per_segment], mean->pose(), covariance);

        // A spatial interpolation point, interpolated in both space and time.
        const double arclength = n[0]->arclength + (n[1]->arclength - n[0]->arclength) / 3.0;
        mean = std::static_pointer_cast<ValuesType>(
            interpolator.interpolate(time, arclength, values, n[0], n[3], n[1], n[4], nullptr, nullptr, posterior.get(), &covariance));
        expectEllipsoidMatches(f.view->mp_covariance_actors[1], mean->pose(), covariance);

        // Fewer interpolation points hide the ellipsoids that are no longer used.
        f.view->adjustSpatialInterpolationPoints(-2);
        f.view->updateVisualization(time, 0.1);
        EXPECT_EQ(f.visibleEllipsoids(), kSpatialDim);
    }

    TEST(ContinuumRobotCovariance, NothingIsDrawnWithoutAnExtractedCovariance)
    {
        RobotFixture f(/*compute_covariance=*/false);
        f.view->toggleCovariance();
        f.view->adjustSpatialInterpolationPoints(+1);
        EXPECT_NO_THROW(f.view->updateVisualization(0.1, 0.1));
        EXPECT_EQ(f.visibleEllipsoids(), 0u);
        EXPECT_TRUE(f.view->mp_backbone_actor->GetVisibility());
    }

    TEST(ContinuumRobotCovariance, HidingTheRobotHidesItsEllipsoids)
    {
        RobotFixture f(/*compute_covariance=*/true);
        f.view->toggleCovariance();
        f.view->updateVisualization(0.1, 0.1);
        ASSERT_EQ(f.visibleEllipsoids(), kSpatialDim);

        f.view->toggleVisibility();
        EXPECT_EQ(f.visibleEllipsoids(), 0u);
        f.view->updateVisualization(0.1, 0.1);
        EXPECT_EQ(f.visibleEllipsoids(), 0u);
    }

    TEST(ContinuumRobotCovariance, EllipsoidsAreOnlyCreatedOnceFirstToggledOn)
    {
        RobotFixture f(/*compute_covariance=*/true, /*prefill_ellipsoids=*/false);
        vtkNew<vtkRenderer> renderer;
        f.view->mp_covariance_renderer = renderer;
        f.view->updateVisualization(0.1, 0.1);
        EXPECT_TRUE(f.view->mp_covariance_actors.empty()) << "a robot never showing its covariance needs no ellipsoid actors";
        EXPECT_EQ(renderer->GetActors()->GetNumberOfItems(), 0);

        f.view->toggleCovariance();
        ASSERT_EQ(f.view->mp_covariance_actors.size(), RobotFixture::kMaxEllipsoids);
        EXPECT_EQ(renderer->GetActors()->GetNumberOfItems(), static_cast<int>(RobotFixture::kMaxEllipsoids));
        for (const auto &actor : f.view->mp_covariance_actors)
            EXPECT_EQ(actor->GetMapper(), f.view->mp_covariance_mapper.Get()) << "every ellipsoid shares the one sphere mapper";
        f.view->updateVisualization(0.1, 0.1);
        EXPECT_EQ(f.visibleEllipsoids(), kSpatialDim);

        f.view->toggleCovariance();
        f.view->toggleCovariance();
        EXPECT_EQ(f.view->mp_covariance_actors.size(), RobotFixture::kMaxEllipsoids) << "toggling again must not create more";
        EXPECT_EQ(renderer->GetActors()->GetNumberOfItems(), static_cast<int>(RobotFixture::kMaxEllipsoids));
    }

    TEST(ContinuumRobotCovariance, LazyEllipsoidsWithoutARendererStillWork)
    {
        // As the other tests' views, which never visualize(): the actors are created but added nowhere.
        RobotFixture f(/*compute_covariance=*/true, /*prefill_ellipsoids=*/false);
        f.view->toggleCovariance();
        ASSERT_EQ(f.view->mp_covariance_actors.size(), RobotFixture::kMaxEllipsoids);
        f.view->updateVisualization(0.1, 0.1);
        EXPECT_EQ(f.visibleEllipsoids(), kSpatialDim);
    }
} // namespace Spacetime
