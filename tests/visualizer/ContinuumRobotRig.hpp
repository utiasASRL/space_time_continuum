#ifndef TESTS_VISUALIZER_CONTINUUMROBOTRIG_HPP
#define TESTS_VISUALIZER_CONTINUUMROBOTRIG_HPP

#include <memory>

#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>
#include <vtkActor.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/visualizer/default/objects/ContinuumRobot.hpp>

/// The robot rig shared by the Default::ContinuumRobot view tests (ContinuumRobotVisualizationTests.cpp, ContinuumRobotSTLTests.cpp).
namespace Spacetime::VisualizerTests
{
    using Pose = gtsam::Pose3;
    using NodeType = Priors::WNOA::Node<Pose>;
    using StateType = Types::ContinuumRobotState<NodeType>;
    using ValueType = Priors::WNOA::NodeValues<Pose>;
    using PriorT = Priors::WNOA::Prior<Pose>;

    constexpr unsigned int kSpatialDim = 4;
    constexpr double kPeriod = 0.1;

    /// Robot + prior with the view's backbone/frame actors set up directly (as
    /// Default::ContinuumRobot::visualize() would), so updateVisualization() can be
    /// exercised without a Qt/VTK Visualizer or render window.
    struct Rig
    {
        std::shared_ptr<Visualization::Default::ContinuumRobot<NodeType>> view;
        std::shared_ptr<StateType> state;
        std::shared_ptr<PriorT> prior;
        gtsam::Values values; // every value created so far, i.e. what a fully caught-up snapshot would hold
        KeyManager km;

        Rig()
        {
            state = std::make_shared<StateType>(1.0, kSpatialDim);
            view = std::make_shared<Visualization::Default::ContinuumRobot<NodeType>>(state);
            ValueType reference;
            reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
            prior = std::make_shared<PriorT>(state, reference);

            view->rebuildBackboneGeometry();
            view->mp_backbone_actor = vtkSmartPointer<vtkActor>::New();
            for (unsigned int i = 0; i < kSpatialDim; i++)
                view->mp_axes.push_back(vtkSmartPointer<vtkActor>::New());
        }

        /// Lets the view interpolate between nodes with the prior's own interpolator, as the demo does.
        void bindInterpolator()
        {
            view->bindInterpolator([this](double time, double arclength, const gtsam::Values &v,
                                          const std::shared_ptr<NodeType> &n00, const std::shared_ptr<NodeType> &n01,
                                          const std::shared_ptr<NodeType> &n10, const std::shared_ptr<NodeType> &n11,
                                          const PosteriorCovariance *posterior, Eigen::MatrixXd *posterior_covariance)
                                   { return prior->interpolator().interpolate(time, arclength, v, n00, n01, n10, n11,
                                                                              nullptr, nullptr, posterior, posterior_covariance); });
        }

        /// Appends one temporal row at `time`, as the start of Estimator::estimate(time) does.
        void addRow(double time)
        {
            gtsam::Values new_values;
            prior->extendState(km, time, values, new_values);
            values.insert(new_values);
        }

        /// Makes updateVisualization() see `snapshot` as the latest published estimate, as the online demo does.
        void publish(double time, const gtsam::Values &snapshot_values)
        {
            auto snapshot = std::make_shared<const EstimatorSnapshot>(EstimatorSnapshot{time, snapshot_values});
            view->bindSnapshotSource([snapshot](double)
                                      { return snapshot; });
        }

        /// Shifts every pose in temporal row `row` so rows are distinguishable by where the backbone is drawn.
        void offsetRow(unsigned int row, gtsam::Values &target, double dz) const
        {
            for (unsigned int i = 0; i < kSpatialDim; i++)
            {
                const auto &node = state->estimation_nodes.at(row * kSpatialDim + i);
                const Pose pose = target.at<Pose>(node->poseKey());
                target.update(node->poseKey(), Pose(pose.rotation(), pose.translation() + gtsam::Point3(0, 0, dz)));
            }
        }

        bool backboneVisible() const { return view->mp_backbone_actor->GetVisibility(); }

        /// Whether every node frame is drawn (true) or none is (false); fails if they disagree.
        bool nodeFramesVisible() const
        {
            const bool first = view->mp_axes.front()->GetVisibility();
            for (const auto &axes : view->mp_axes)
                EXPECT_EQ(static_cast<bool>(axes->GetVisibility()), first);
            return first;
        }

        /// One published row at t = 0, drawn.
        void drawOneRow()
        {
            addRow(0.0);
            publish(0.0, values);
            view->updateVisualization(0.0, 0.0);
        }

        /// z of the drawn backbone's first point, compared against the base pose of the row expected to be drawn.
        double drawnBaseZ() const
        {
            double point[3];
            view->mp_backbone_points->GetPoint(0, point);
            return point[2];
        }

        double rowBaseZ(unsigned int row, const gtsam::Values &source) const
        {
            return source.at<Pose>(state->estimation_nodes.at(row * kSpatialDim)->poseKey()).z();
        }
    };
} // namespace Spacetime::VisualizerTests

#endif // TESTS_VISUALIZER_CONTINUUMROBOTRIG_HPP
