#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/PosteriorCovariance.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/visualizer/PlotData.hpp>

namespace Spacetime
{
    using namespace Visualization;

    namespace
    {
        using Pose = gtsam::Pose3;
        using WNOANode = Priors::WNOA::Node<Pose>;
        using WNOVNode = Priors::WNOV::Node<Pose>;
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

        /// A pose plus an extra 3-vector, standing in for a future node type with an actuation state.
        struct PoseActuationNode : public Types::NodeBase
        {
            PoseActuationNode(KeyManager &km, double arclength, double time) : Types::NodeBase(km, arclength, time)
            {
                pose_key = registerVariable<Pose>(km, "pose", 'p');
                actuation_key = registerVariable<gtsam::Vector3>(km, "actuation", 'a');
            }
            gtsam::Key pose_key, actuation_key;
        };

        /// A node holding a type with no built-in plotter.
        struct UnitNode : public Types::NodeBase
        {
            UnitNode(KeyManager &km, double arclength, double time) : Types::NodeBase(km, arclength, time)
            {
                direction_key = registerVariable<gtsam::Unit3>(km, "direction", 'u');
            }
            gtsam::Key direction_key;
        };

        /// Row-major grid of nodes (row = time, column = arclength), as ContinuumRobotState lays them out.
        template <typename NodeT>
        struct Grid
        {
            KeyManager km;
            unsigned int spatial_dim;
            std::vector<std::shared_ptr<NodeT>> nodes;

            explicit Grid(unsigned int spatial_dim) : spatial_dim(spatial_dim) {}

            void addRow(double time)
            {
                for (unsigned int i = 0; i < spatial_dim; i++)
                    nodes.push_back(std::make_shared<NodeT>(km, static_cast<double>(i) / (spatial_dim - 1), time));
            }
            const std::shared_ptr<NodeT> &at(unsigned int row, unsigned int spatial) const { return nodes[row * spatial_dim + spatial]; }
        };

        Pose poseAtHeight(double z) { return Pose(gtsam::Rot3::Ypr(0.1, 0.2, 0.3), gtsam::Point3(0.0, 0.0, z)); }

        /// Values for every WNOA node in rows [first_row, last_row], pose z = time + arclength.
        gtsam::Values wnoaValues(const Grid<WNOANode> &grid, unsigned int first_row, unsigned int last_row, double offset = 0.0)
        {
            gtsam::Values values;
            for (unsigned int row = first_row; row <= last_row; row++)
                for (unsigned int i = 0; i < grid.spatial_dim; i++)
                {
                    const auto &node = grid.at(row, i);
                    values.insert(node->poseKey(), poseAtHeight(node->time + node->arclength + offset));
                    values.insert(node->strainKey(), gtsam::Vector6(gtsam::Vector6::Constant(node->arclength)));
                    values.insert(node->velocityKey(), gtsam::Vector6(gtsam::Vector6::Constant(node->time)));
                }
            return values;
        }

        EstimatorSnapshot snapshotOf(double time, gtsam::Values values, std::shared_ptr<const PosteriorCovariance> covariance = nullptr)
        {
            return EstimatorSnapshot{time, std::move(values), std::move(covariance)};
        }

        const PlotGroup *findGroup(const std::vector<PlotGroup> &groups, const std::string &name)
        {
            for (const auto &group : groups)
                if (group.name == name)
                    return &group;
            return nullptr;
        }

        /// First-order 1-sigma of each poseComponents() entry, by central differences over the right perturbation T * Exp(delta).
        std::array<double, 6> finiteDifferenceSigmas(const Pose &pose, const Eigen::Matrix<double, 6, 6> &covariance)
        {
            constexpr double h = 1e-6;
            Eigen::Matrix<double, 6, 6> jacobian;
            for (int j = 0; j < 6; j++)
            {
                gtsam::Vector6 delta = gtsam::Vector6::Zero();
                delta(j) = h;
                const auto plus = poseComponents(pose.compose(Pose::Expmap(delta)));
                const auto minus = poseComponents(pose.compose(Pose::Expmap(-delta)));
                for (int i = 0; i < 6; i++)
                    jacobian(i, j) = (plus[i] - minus[i]) / (2 * h);
            }
            const Eigen::Matrix<double, 6, 6> propagated = jacobian * covariance * jacobian.transpose();
            std::array<double, 6> sigmas;
            for (int i = 0; i < 6; i++)
                sigmas[i] = std::sqrt(propagated(i, i));
            return sigmas;
        }

        Types::MeasurementQueue::pMeasurementType poseMeasurement(double arclength, double time, double z)
        {
            return std::make_shared<Types::MeasurementBase>(arclength, time, poseAtHeight(z).matrix(), Eigen::MatrixXd::Identity(6, 6));
        }
    } // namespace

    // ---------------------------------------------------------------- plotters

    TEST(PlotData, PoseComponentsOfIdentityAreZero)
    {
        const auto components = poseComponents(Pose());
        for (double c : components)
            EXPECT_DOUBLE_EQ(c, 0.0);
    }

    TEST(PlotData, PoseComponentsArePositionThenZyxRollPitchYaw)
    {
        const Pose pose(gtsam::Rot3::Ypr(0.3, -0.2, 0.1), gtsam::Point3(1.0, 2.0, 3.0));
        const auto c = poseComponents(pose);
        EXPECT_NEAR(c[0], 1.0, 1e-12);
        EXPECT_NEAR(c[1], 2.0, 1e-12);
        EXPECT_NEAR(c[2], 3.0, 1e-12);
        EXPECT_NEAR(c[3], 0.1, 1e-12);  // roll
        EXPECT_NEAR(c[4], -0.2, 1e-12); // pitch
        EXPECT_NEAR(c[5], 0.3, 1e-12);  // yaw
    }

    TEST(PlotData, PoseSigmasMatchFirstOrderPropagation)
    {
        const Pose pose(gtsam::Rot3::Ypr(0.7, -0.4, 1.1), gtsam::Point3(0.5, -0.3, 0.2));
        Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Random();
        const Eigen::Matrix<double, 6, 6> covariance = 1e-3 * (A * A.transpose() + Eigen::Matrix<double, 6, 6>::Identity());

        const auto analytic = poseSigmas(pose, covariance);
        const auto reference = finiteDifferenceSigmas(pose, covariance);
        for (int i = 0; i < 6; i++)
            EXPECT_NEAR(analytic[i], reference[i], 1e-6) << "component " << i;
    }

    TEST(PlotData, PoseSigmasAreNaNWithoutACovariance)
    {
        for (double s : poseSigmas(Pose(), Eigen::MatrixXd()))
            EXPECT_TRUE(std::isnan(s));
    }

    TEST(PlotData, PoseAngleSigmasAreNaNAtGimbalLockButPositionSigmasAreNot)
    {
        const Pose pose(gtsam::Rot3::Ypr(0.0, M_PI / 2, 0.0), gtsam::Point3());
        const auto sigmas = poseSigmas(pose, Eigen::MatrixXd::Identity(6, 6) * 1e-4);
        for (int i = 0; i < 3; i++)
            EXPECT_TRUE(std::isfinite(sigmas[i]));
        EXPECT_TRUE(std::isnan(sigmas[3]));
        EXPECT_TRUE(std::isnan(sigmas[5]));
    }

    TEST(PlotData, BuiltInPlottersCoverVectorsScalarsAndPose2)
    {
        PlotterRegistry registry;
        gtsam::Values values;
        values.insert(1, gtsam::Vector3(1.0, 2.0, 3.0));
        values.insert(2, 4.5);
        values.insert(3, gtsam::Pose2(1.0, 2.0, 0.5));
        values.insert(4, gtsam::Vector(gtsam::Vector::Constant(4, 7.0)));

        const auto *vec = registry.find(typeid(gtsam::Vector3));
        ASSERT_NE(vec, nullptr);
        EXPECT_EQ(vec->components(values, 1), (std::vector<double>{1.0, 2.0, 3.0}));
        EXPECT_EQ(vec->labels("actuation", values, 1), (std::vector<std::string>{"actuation[0]", "actuation[1]", "actuation[2]"}));
        const auto sigmas = vec->sigmas(values, 1, Eigen::Vector3d(4.0, 9.0, 16.0).asDiagonal());
        EXPECT_EQ(sigmas, (std::vector<double>{2.0, 3.0, 4.0}));

        const auto *scalar = registry.find(typeid(double));
        ASSERT_NE(scalar, nullptr);
        EXPECT_EQ(scalar->components(values, 2), (std::vector<double>{4.5}));

        const auto *pose2 = registry.find(typeid(gtsam::Pose2));
        ASSERT_NE(pose2, nullptr);
        EXPECT_EQ(pose2->components(values, 3), (std::vector<double>{1.0, 2.0, 0.5}));

        const auto *dynamic = registry.find(typeid(gtsam::Vector));
        ASSERT_NE(dynamic, nullptr);
        EXPECT_EQ(dynamic->components(values, 4).size(), 4u);
    }

    TEST(PlotData, VectorSigmasAreNaNWithoutACovariance)
    {
        PlotterRegistry registry;
        gtsam::Values values;
        values.insert(1, gtsam::Vector3(1.0, 2.0, 3.0));
        for (double s : registry.find(typeid(gtsam::Vector3))->sigmas(values, 1, Eigen::MatrixXd()))
            EXPECT_TRUE(std::isnan(s));
    }

    TEST(PlotData, UnregisteredTypeHasNoPlotterAndCanBeRegistered)
    {
        PlotterRegistry registry;
        EXPECT_EQ(registry.find(typeid(gtsam::Unit3)), nullptr);

        registry.registerPlotter<gtsam::Unit3>(
            [](const std::string &name, const gtsam::Unit3 &)
            { return std::vector<std::string>{name + ".x", name + ".y", name + ".z"}; },
            [](const gtsam::Unit3 &u)
            { const auto p = u.point3(); return std::vector<double>{p.x(), p.y(), p.z()}; },
            [](const gtsam::Unit3 &, const Eigen::MatrixXd &)
            { return std::vector<double>(3, kNaN); });

        gtsam::Values values;
        values.insert(1, gtsam::Unit3(0.0, 0.0, 1.0));
        const auto *plotter = registry.find(typeid(gtsam::Unit3));
        ASSERT_NE(plotter, nullptr);
        EXPECT_EQ(plotter->components(values, 1), (std::vector<double>{0.0, 0.0, 1.0}));
    }

    // ------------------------------------------------------------ dynamic groups

    TEST(NodeHistory, WNOANodeGivesPoseStrainAndVelocityGroups)
    {
        Grid<WNOANode> grid(3);
        grid.addRow(1.0);
        NodeHistory history;
        ASSERT_TRUE(history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(1.0, wnoaValues(grid, 0, 0))));

        const auto groups = history.sliceAtTime(0);
        ASSERT_EQ(groups.size(), 3u);
        // NodeBase::names() order (alphabetical).
        EXPECT_EQ(groups[0].name, "pose");
        EXPECT_EQ(groups[1].name, "strain");
        EXPECT_EQ(groups[2].name, "velocity");
        EXPECT_EQ(groups[0].labels, (std::vector<std::string>{"x", "y", "z", "roll", "pitch", "yaw"}));
        for (const auto &group : groups)
        {
            EXPECT_TRUE(group.supported);
            ASSERT_EQ(group.series.size(), 6u);
            EXPECT_EQ(group.series[0].x.size(), 3u);
        }
    }

    TEST(NodeHistory, WNOVNodeGivesOnlyAPoseGroup)
    {
        Grid<WNOVNode> grid(2);
        grid.addRow(1.0);
        gtsam::Values values;
        for (const auto &node : grid.nodes)
            values.insert(node->poseKey(), poseAtHeight(node->arclength));
        NodeHistory history;
        history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(1.0, values));

        const auto groups = history.sliceAtTime(0);
        ASSERT_EQ(groups.size(), 1u);
        EXPECT_EQ(groups[0].name, "pose");
        EXPECT_EQ(groups[0].series.size(), 6u);
    }

    TEST(NodeHistory, ExtraNodeVariableGetsItsOwnGroup)
    {
        Grid<PoseActuationNode> grid(2);
        grid.addRow(1.0);
        gtsam::Values values;
        for (const auto &node : grid.nodes)
        {
            values.insert(node->pose_key, poseAtHeight(node->arclength));
            values.insert(node->actuation_key, gtsam::Vector3(1.0, 2.0, node->arclength));
        }
        NodeHistory history;
        history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(1.0, values));

        const auto groups = history.sliceAtTime(0);
        ASSERT_EQ(groups.size(), 2u);
        const auto *actuation = findGroup(groups, "actuation");
        const auto *pose = findGroup(groups, "pose");
        ASSERT_NE(actuation, nullptr);
        ASSERT_NE(pose, nullptr);
        EXPECT_EQ(pose->series.size(), 6u);
        ASSERT_EQ(actuation->series.size(), 3u);
        EXPECT_EQ(actuation->series[2].x, (std::vector<double>{0.0, 1.0}));
        EXPECT_EQ(actuation->series[2].mean, (std::vector<double>{0.0, 1.0}));
    }

    TEST(NodeHistory, UnsupportedVariableIsReportedNotThrown)
    {
        Grid<UnitNode> grid(2);
        grid.addRow(1.0);
        gtsam::Values values;
        for (const auto &node : grid.nodes)
            values.insert(node->direction_key, gtsam::Unit3(0.0, 0.0, 1.0));
        NodeHistory history;
        EXPECT_NO_THROW(history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(1.0, values)));

        const auto groups = history.sliceAtTime(0);
        ASSERT_EQ(groups.size(), 1u);
        EXPECT_EQ(groups[0].name, "direction");
        EXPECT_FALSE(groups[0].supported);
        EXPECT_TRUE(groups[0].series.empty());
    }

    TEST(NodeHistory, RegisteredPlotterIsUsedForANewType)
    {
        Grid<UnitNode> grid(2);
        grid.addRow(1.0);
        gtsam::Values values;
        for (const auto &node : grid.nodes)
            values.insert(node->direction_key, gtsam::Unit3(0.0, 0.0, 1.0));
        NodeHistory history;
        history.plotters().registerPlotter<gtsam::Unit3>(
            [](const std::string &name, const gtsam::Unit3 &)
            { return std::vector<std::string>{name + ".z"}; },
            [](const gtsam::Unit3 &u)
            { return std::vector<double>{u.point3().z()}; },
            [](const gtsam::Unit3 &, const Eigen::MatrixXd &)
            { return std::vector<double>{kNaN}; });
        history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(1.0, values));

        const auto groups = history.sliceAtTime(0);
        ASSERT_EQ(groups.size(), 1u);
        EXPECT_TRUE(groups[0].supported);
        ASSERT_EQ(groups[0].series.size(), 1u);
        EXPECT_EQ(groups[0].series[0].mean, (std::vector<double>{1.0, 1.0}));
    }

    // ----------------------------------------------------------------- history

    TEST(NodeHistory, EmptySnapshotAndNoNodesChangeNothing)
    {
        Grid<WNOANode> grid(2);
        NodeHistory history;
        EXPECT_FALSE(history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(0.0, gtsam::Values())));
        grid.addRow(1.0);
        EXPECT_FALSE(history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(1.0, gtsam::Values())));
        EXPECT_EQ(history.numRows(), 0u);
        EXPECT_TRUE(history.sliceAtArclength(0, 10.0).empty());
        EXPECT_TRUE(history.sliceAtTime(0).empty());
        EXPECT_FALSE(history.latestRowAtOrBefore(10.0).has_value());
    }

    TEST(NodeHistory, MarginalizedRowsKeepTheirLastValue)
    {
        Grid<WNOANode> grid(2);
        grid.addRow(1.0);
        grid.addRow(2.0);
        NodeHistory history;
        history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(2.0, wnoaValues(grid, 0, 1)));

        // Row 0 leaves the window; row 2 joins.
        grid.addRow(3.0);
        history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(3.0, wnoaValues(grid, 1, 2, /*offset=*/10.0)));

        const auto groups = history.sliceAtArclength(0, 10.0);
        const auto *pose = findGroup(groups, "pose");
        ASSERT_NE(pose, nullptr);
        const auto &z = pose->series[2];
        EXPECT_EQ(z.x, (std::vector<double>{1.0, 2.0, 3.0}));
        EXPECT_NEAR(z.mean[0], 1.0, 1e-12);  // frozen from the first snapshot
        EXPECT_NEAR(z.mean[1], 12.0, 1e-12); // refreshed while still in the window
        EXPECT_NEAR(z.mean[2], 13.0, 1e-12);
    }

    TEST(NodeHistory, RowsNewerThanTheSnapshotDoNotStopIngestion)
    {
        // Mid-solve (online) or after rewinding (offline), the newest rows are not in the
        // snapshot yet; the rows that are must still be ingested.
        Grid<WNOANode> grid(2);
        grid.addRow(1.0);
        grid.addRow(2.0);
        grid.addRow(3.0);
        NodeHistory history;
        EXPECT_TRUE(history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(2.0, wnoaValues(grid, 0, 1))));
        EXPECT_EQ(history.sliceAtArclength(1, 10.0)[0].series[0].x, (std::vector<double>{1.0, 2.0}));
    }

    TEST(NodeHistory, ArclengthSliceIsTimeOrderedAndTruncated)
    {
        Grid<WNOANode> grid(3);
        for (double t : {1.0, 2.0, 3.0})
            grid.addRow(t);
        NodeHistory history;
        history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(3.0, wnoaValues(grid, 0, 2)));

        const auto groups = history.sliceAtArclength(2, 2.5);
        const auto *pose = findGroup(groups, "pose");
        ASSERT_NE(pose, nullptr);
        EXPECT_EQ(pose->series[2].x, (std::vector<double>{1.0, 2.0}));
        EXPECT_NEAR(pose->series[2].mean[1], 3.0, 1e-12); // time 2 + arclength 1
    }

    TEST(NodeHistory, TimeSliceIsArclengthOrdered)
    {
        Grid<WNOANode> grid(3);
        grid.addRow(1.0);
        grid.addRow(2.0);
        NodeHistory history;
        history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(2.0, wnoaValues(grid, 0, 1)));

        const auto groups = history.sliceAtTime(1);
        const auto *strain = findGroup(groups, "strain");
        ASSERT_NE(strain, nullptr);
        EXPECT_EQ(strain->series[0].x, (std::vector<double>{0.0, 0.5, 1.0}));
        EXPECT_EQ(strain->series[0].mean, (std::vector<double>{0.0, 0.5, 1.0}));
    }

    TEST(NodeHistory, LatestRowAtOrBefore)
    {
        Grid<WNOANode> grid(2);
        for (double t : {1.0, 2.0, 3.0})
            grid.addRow(t);
        NodeHistory history;
        history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(3.0, wnoaValues(grid, 0, 2)));
        EXPECT_FALSE(history.latestRowAtOrBefore(0.5).has_value());
        EXPECT_EQ(history.latestRowAtOrBefore(1.0), 0u);
        EXPECT_EQ(history.latestRowAtOrBefore(2.9), 1u);
        EXPECT_EQ(history.latestRowAtOrBefore(30.0), 2u);
    }

    TEST(NodeHistory, BoundsAreNaNWithoutCovarianceAndThreeSigmaWithIt)
    {
        Grid<WNOANode> grid(2);
        grid.addRow(1.0);
        NodeHistory history;
        history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(1.0, wnoaValues(grid, 0, 0)));
        auto strain = *findGroup(history.sliceAtTime(0), "strain");
        EXPECT_TRUE(std::isnan(strain.series[0].lower[0]));
        EXPECT_TRUE(std::isnan(strain.series[0].upper[0]));

        auto covariance = std::make_shared<PosteriorCovariance>();
        for (const auto &node : grid.nodes)
        {
            covariance->insert(node->poseKey(), node->poseKey(), Eigen::MatrixXd::Identity(6, 6) * 1e-4);
            covariance->insert(node->strainKey(), node->strainKey(), Eigen::MatrixXd::Identity(6, 6) * 4.0);
        }
        history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(1.0, wnoaValues(grid, 0, 0), covariance));
        strain = *findGroup(history.sliceAtTime(0), "strain");
        EXPECT_NEAR(strain.series[0].lower[1], 1.0 - 6.0, 1e-12); // mean 1 (arclength), sigma 2
        EXPECT_NEAR(strain.series[0].upper[1], 1.0 + 6.0, 1e-12);
        // Velocity has no extracted block: still NaN.
        const auto velocity = *findGroup(history.sliceAtTime(0), "velocity");
        EXPECT_TRUE(std::isnan(velocity.series[0].lower[0]));
    }

    TEST(NodeHistory, RevisionChangesOnlyWhenDataChanges)
    {
        Grid<WNOANode> grid(2);
        grid.addRow(1.0);
        NodeHistory history;
        const auto before = history.revision();
        history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(1.0, wnoaValues(grid, 0, 0)));
        EXPECT_NE(history.revision(), before);
        const auto after = history.revision();
        history.ingest(grid.nodes, grid.spatial_dim, snapshotOf(1.0, gtsam::Values()));
        EXPECT_EQ(history.revision(), after);
    }

    // ------------------------------------------------------------ ground truth

    TEST(PlotGroundTruth, AtArclengthUsesTheClosestSensorWithinTolerance)
    {
        const std::vector<std::vector<Types::MeasurementQueue::pMeasurementType>> sensors = {
            {poseMeasurement(0.5, 1.0, 1.0), poseMeasurement(0.5, 2.0, 2.0), poseMeasurement(0.5, 3.0, 3.0)},
            {poseMeasurement(0.6, 1.0, 9.0)},
            {poseMeasurement(0.9, 1.0, 9.0)},
        };
        const auto gt = poseGroundTruthAtArclength(sensors, 0.52, 0.1, 2.5);
        EXPECT_EQ(gt.x, (std::vector<double>{1.0, 2.0}));
        ASSERT_EQ(gt.components.size(), 6u);
        EXPECT_NEAR(gt.components[2][1], 2.0, 1e-12);

        EXPECT_TRUE(poseGroundTruthAtArclength(sensors, 0.2, 0.1, 10.0).x.empty());
    }

    TEST(PlotGroundTruth, AtTimeTakesEachSensorsNearestMeasurementWithinTolerance)
    {
        const std::vector<std::vector<Types::MeasurementQueue::pMeasurementType>> sensors = {
            {poseMeasurement(1.0, 1.0, 1.0), poseMeasurement(1.0, 2.0, 2.0)},
            {poseMeasurement(0.0, 1.9, 5.0), poseMeasurement(0.0, 2.04, 6.0)},
            {poseMeasurement(0.5, 5.0, 7.0)}, // too far in time
        };
        const auto gt = poseGroundTruthAtTime(sensors, 2.0, 0.05);
        EXPECT_EQ(gt.x, (std::vector<double>{0.0, 1.0})); // ordered by arclength
        EXPECT_NEAR(gt.components[2][0], 6.0, 1e-12);
        EXPECT_NEAR(gt.components[2][1], 2.0, 1e-12);
    }

    TEST(PlotGroundTruth, NoSensorsGiveNoPoints)
    {
        EXPECT_TRUE(poseGroundTruthAtTime({}, 1.0, 0.1).x.empty());
        EXPECT_TRUE(poseGroundTruthAtArclength({}, 1.0, 0.1, 1.0).x.empty());
    }

    // --------------------------------------------------------------- selection

    TEST(PlotSelection, StartsAtTheTipAndClampsSteps)
    {
        PlotSelection selection(4);
        EXPECT_EQ(selection.spatialIndex(), 3u);
        EXPECT_FALSE(selection.step(+1));
        EXPECT_TRUE(selection.step(-2));
        EXPECT_EQ(selection.spatialIndex(), 1u);
        EXPECT_TRUE(selection.step(-5));
        EXPECT_EQ(selection.spatialIndex(), 0u);
        EXPECT_FALSE(selection.select(7));
        EXPECT_TRUE(selection.select(2));
        EXPECT_EQ(selection.spatialIndex(), 2u);
    }

    TEST(PlotSelection, ModeTogglesAndBumpsRevision)
    {
        PlotSelection selection(2);
        EXPECT_EQ(selection.mode(), PlotMode::TimeAtArclength);
        const auto revision = selection.revision();
        selection.toggleMode();
        EXPECT_EQ(selection.mode(), PlotMode::ArclengthAtTime);
        EXPECT_NE(selection.revision(), revision);
        selection.toggleMode();
        EXPECT_EQ(selection.mode(), PlotMode::TimeAtArclength);
    }

    // ------------------------------------------------------------- view range

    TEST(PlotRange, FitIsEmptyWithoutFiniteData)
    {
        EXPECT_FALSE(fitRange(PlotSeries{}, {}, {}).has_value());
        EXPECT_FALSE(fitRange(PlotSeries{{kNaN}, {1.0}, {kNaN}, {kNaN}}, {}, {}).has_value());
    }

    TEST(PlotRange, FitCoversMeanBoundsAndGroundTruthWithValuePadding)
    {
        const PlotSeries series{{0.0, 1.0}, {0.0, 1.0}, {-1.0, kNaN}, {1.0, kNaN}};
        const auto range = fitRange(series, {2.0}, {3.0});
        ASSERT_TRUE(range.has_value());
        EXPECT_DOUBLE_EQ(range->x_min, 0.0);
        EXPECT_DOUBLE_EQ(range->x_max, 2.0);
        // Values span [-1, 3], padded by 10% of the span on each side.
        EXPECT_DOUBLE_EQ(range->y_min, -1.4);
        EXPECT_DOUBLE_EQ(range->y_max, 3.4);
    }

    TEST(PlotRange, FitPadsASinglePoint)
    {
        const auto range = fitRange(PlotSeries{{1.0}, {2.0}, {kNaN}, {kNaN}}, {}, {});
        ASSERT_TRUE(range.has_value());
        EXPECT_LT(range->x_min, 1.0);
        EXPECT_GT(range->x_max, 1.0);
        EXPECT_LT(range->y_min, 2.0);
        EXPECT_GT(range->y_max, 2.0);
    }

    TEST(PlotRange, ZoomKeepsTheAnchorFixedAndScalesEachAxisIndependently)
    {
        const PlotRange range{0.0, 10.0, -5.0, 5.0};
        const PlotRange both = zoomRange(range, 2.0, 1.0, 0.5, 0.5);
        EXPECT_DOUBLE_EQ(both.x_min, 1.0);
        EXPECT_DOUBLE_EQ(both.x_max, 6.0);
        EXPECT_DOUBLE_EQ(both.y_min, -2.0);
        EXPECT_DOUBLE_EQ(both.y_max, 3.0);

        const PlotRange x_only = zoomRange(range, 2.0, 1.0, 2.0, 1.0);
        EXPECT_DOUBLE_EQ(x_only.x_min, -2.0);
        EXPECT_DOUBLE_EQ(x_only.x_max, 18.0);
        EXPECT_DOUBLE_EQ(x_only.y_min, range.y_min);
        EXPECT_DOUBLE_EQ(x_only.y_max, range.y_max);
    }

    TEST(PlotRange, ExtremeZoomStaysFiniteAndOrdered)
    {
        PlotRange range{0.0, 1.0, 0.0, 1.0};
        for (int i = 0; i < 2000; i++)
            range = zoomRange(range, 0.5, 0.5, 0.5, 0.5);
        EXPECT_LT(range.x_min, range.x_max);
        EXPECT_LT(range.y_min, range.y_max);
        for (int i = 0; i < 4000; i++)
            range = zoomRange(range, 0.5, 0.5, 2.0, 2.0);
        EXPECT_TRUE(std::isfinite(range.x_min) && std::isfinite(range.x_max));
        EXPECT_TRUE(std::isfinite(range.y_min) && std::isfinite(range.y_max));
    }

    TEST(PlotRange, PanShiftsBothEnds)
    {
        const PlotRange panned = panRange(PlotRange{0.0, 10.0, -5.0, 5.0}, 1.5, -2.0);
        EXPECT_DOUBLE_EQ(panned.x_min, 1.5);
        EXPECT_DOUBLE_EQ(panned.x_max, 11.5);
        EXPECT_DOUBLE_EQ(panned.y_min, -7.0);
        EXPECT_DOUBLE_EQ(panned.y_max, 3.0);
    }
} // namespace Spacetime
