#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>
#include <vtkActor.h>
#include <vtkCubeSource.h>
#include <vtkMapper.h>
#include <vtkMatrix4x4.h>
#include <vtkPolyData.h>
#include <vtkRenderer.h>
#include <vtkSTLWriter.h>
#include <vtkSmartPointer.h>

#include <spacetime/visualizer/default/utilities.hpp>
#include "ContinuumRobotRig.hpp"

namespace Spacetime
{
    using namespace VisualizerTests;

    namespace
    {
        namespace fs = std::filesystem;

        /// A temporary directory holding a binary STL box of extents (0.2, 0.4, 0.6), centred at (0.1, -0.2, 0.3), removed on destruction.
        struct STLFixture
        {
            fs::path directory, box;

            STLFixture()
            {
                directory = fs::temp_directory_path() / ("spacetime_stl_tests_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                                                         ::testing::UnitTest::GetInstance()->current_test_info()->name());
                fs::create_directories(directory);
                box = directory / "box.stl";
                vtkSmartPointer<vtkCubeSource> cube = vtkSmartPointer<vtkCubeSource>::New();
                cube->SetXLength(0.2);
                cube->SetYLength(0.4);
                cube->SetZLength(0.6);
                cube->SetCenter(0.1, -0.2, 0.3);
                vtkSmartPointer<vtkSTLWriter> writer = vtkSmartPointer<vtkSTLWriter>::New();
                writer->SetInputConnection(cube->GetOutputPort());
                writer->SetFileName(box.c_str());
                writer->SetFileTypeToBinary();
                writer->Write();
            }

            ~STLFixture() { fs::remove_all(directory); }

            /// A file holding `content` (e.g. not an STL at all).
            std::string file(const std::string &name, const std::string &content) const
            {
                const fs::path path = directory / name;
                std::ofstream(path) << content;
                return path.string();
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

        /// The actor's user matrix, or nullopt if it has none.
        std::optional<Eigen::Matrix4d> userMatrix(vtkActor *actor)
        {
            if (!actor->GetUserMatrix())
                return std::nullopt;
            return toEigen(actor->GetUserMatrix());
        }

        vtkPolyData *meshOf(vtkActor *actor) { return vtkPolyData::SafeDownCast(actor->GetMapper()->GetInput()); }

        /// Expects `actor` to be shown at the view's poseAt(time, s, estimate_time).
        void expectDrawnAtPoseAt(const Rig &rig, vtkActor *actor, double time, double s, double estimate_time)
        {
            const auto expected = rig.view->poseAt(time, s, estimate_time);
            ASSERT_TRUE(expected.has_value());
            EXPECT_TRUE(actor->GetVisibility());
            const auto drawn = userMatrix(actor);
            ASSERT_TRUE(drawn.has_value());
            EXPECT_TRUE(drawn->isApprox(*expected, 1e-12)) << "drawn\n"
                                                           << *drawn << "\nexpected\n"
                                                           << *expected;
        }

        const gtsam::Pose3 kOffset(gtsam::Rot3::RzRyRx(0.3, -0.2, 0.5), gtsam::Point3(0.05, -0.1, 0.2));
    } // namespace

    TEST(STLMesh, LoadsTheMeshScaled)
    {
        STLFixture fixture;
        const auto mesh = Visualization::Default::Utilities::loadSTLMesh(fixture.box.string(), 0.5);
        ASSERT_GT(mesh->GetNumberOfPoints(), 0);
        double bounds[6];
        mesh->GetBounds(bounds);
        const double expected[6] = {0.0, 0.1, -0.2, 0.0, 0.0, 0.3}; // the box's bounds, halved
        for (int i = 0; i < 6; i++)
            EXPECT_NEAR(bounds[i], expected[i], 1e-6) << i;
    }

    TEST(STLMesh, MissingOrInvalidFilesThrow)
    {
        STLFixture fixture;
        using Visualization::Default::Utilities::loadSTLMesh;
        EXPECT_THROW(loadSTLMesh((fixture.directory / "missing.stl").string()), std::runtime_error);
        EXPECT_THROW(loadSTLMesh(fixture.directory.string()), std::runtime_error); // a directory
        EXPECT_THROW(loadSTLMesh(fixture.file("empty.stl", "")), std::runtime_error);
        EXPECT_THROW(loadSTLMesh(fixture.file("garbage.stl", "solid nothing\nendsolid nothing\n")), std::runtime_error);
    }

    TEST(ContinuumRobotSTL, AddSTLRejectsInvalidArguments)
    {
        STLFixture fixture;
        Rig rig;
        const std::string box = fixture.box.string();
        EXPECT_THROW(rig.view->addSTL((fixture.directory / "missing.stl").string(), 0.5), std::runtime_error);
        EXPECT_THROW(rig.view->addSTL(box, -0.01), std::invalid_argument);
        EXPECT_THROW(rig.view->addSTL(box, rig.state->length + 0.01), std::invalid_argument);
        EXPECT_THROW(rig.view->addSTL(box, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
        for (const double scale : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
            EXPECT_THROW(rig.view->addSTL(box, 0.5, scale), std::invalid_argument) << scale;
        EXPECT_TRUE(rig.view->mp_meshes.empty()) << "nothing is attached by a rejected call";

        EXPECT_NO_THROW(rig.view->addSTL(box, 0.0));
        EXPECT_NO_THROW(rig.view->addSTL(box, rig.state->length));
        EXPECT_EQ(rig.view->mp_meshes.size(), 2u);
    }

    // The scale and offset are applied once, to the mesh itself, so drawing it only moves its origin.
    TEST(ContinuumRobotSTL, ScaleAndOffsetAreBakedIntoTheMesh)
    {
        STLFixture fixture;
        Rig rig;
        constexpr double kScale = 2.5;
        const auto actor = rig.view->addSTL(fixture.box.string(), 0.5, kScale, kOffset);
        ASSERT_TRUE(actor);
        EXPECT_FALSE(actor->GetVisibility()) << "hidden until drawn";
        EXPECT_FALSE(userMatrix(actor).has_value() && !userMatrix(actor)->isIdentity());

        const auto original = Visualization::Default::Utilities::loadSTLMesh(fixture.box.string());
        vtkPolyData *baked = meshOf(actor);
        ASSERT_NE(baked, nullptr);
        ASSERT_EQ(baked->GetNumberOfPoints(), original->GetNumberOfPoints());
        for (vtkIdType i = 0; i < original->GetNumberOfPoints(); i++)
        {
            double p[3], q[3];
            original->GetPoint(i, p);
            baked->GetPoint(i, q);
            const gtsam::Point3 expected = kOffset.transformFrom(gtsam::Point3(kScale * p[0], kScale * p[1], kScale * p[2]));
            EXPECT_TRUE(Eigen::Vector3d(q[0], q[1], q[2]).isApprox(expected, 1e-5)) << i;
        }
    }

    TEST(ContinuumRobotSTL, MeshFollowsThePoseAtItsArclength)
    {
        STLFixture fixture;
        Rig rig;
        rig.bindInterpolator();
        const double on_node = rig.state->arclengths[1], between = 0.5, tip = rig.state->length;
        const auto at_node = rig.view->addSTL(fixture.box.string(), on_node, 1.0, kOffset);
        const auto at_between = rig.view->addSTL(fixture.box.string(), between);
        const auto at_tip = rig.view->addSTL(fixture.box.string(), tip);
        rig.addRow(0.0);
        rig.publish(0.0, rig.values);

        rig.view->updateVisualization(0.0, 0.0);

        expectDrawnAtPoseAt(rig, at_node, 0.0, on_node, 0.0);
        expectDrawnAtPoseAt(rig, at_between, 0.0, between, 0.0);
        expectDrawnAtPoseAt(rig, at_tip, 0.0, tip, 0.0);
        // The offset is in the mesh, not the matrix: the node's mesh is drawn at the node itself.
        const auto node = rig.state->getClosestShiftedNode(on_node, 0.0, -1, -1);
        EXPECT_TRUE(userMatrix(at_node)->isApprox(rig.values.at<Pose>(node->poseKey()).matrix(), 1e-9));
        // Straight along z: between nodes, z is the arclength.
        EXPECT_NEAR((*userMatrix(at_between))(2, 3), between, 1e-6);
    }

    TEST(ContinuumRobotSTL, MeshMovesWithTheEstimate)
    {
        STLFixture fixture;
        Rig rig;
        rig.bindInterpolator();
        const auto actor = rig.view->addSTL(fixture.box.string(), 0.5);
        rig.addRow(0.0);
        rig.publish(0.0, rig.values);
        rig.view->updateVisualization(0.0, 0.0);
        vtkMatrix4x4 *matrix = actor->GetUserMatrix();
        ASSERT_NE(matrix, nullptr);

        gtsam::Values moved = rig.values;
        rig.offsetRow(0, moved, 0.5);
        rig.publish(0.0, moved);
        rig.view->updateVisualization(0.0, 0.0);
        EXPECT_EQ(actor->GetUserMatrix(), matrix) << "updated in place";
        expectDrawnAtPoseAt(rig, actor, 0.0, 0.5, 0.0);
        EXPECT_NEAR(matrix->GetElement(2, 3), 1.0, 1e-6);
    }

    TEST(ContinuumRobotSTL, MeshFollowsTemporalInterpolation)
    {
        STLFixture fixture;
        Rig rig;
        rig.bindInterpolator();
        const auto actor = rig.view->addSTL(fixture.box.string(), 0.5);
        rig.addRow(0.0);
        rig.addRow(kPeriod);
        gtsam::Values shifted = rig.values;
        rig.offsetRow(1, shifted, 0.2);
        rig.publish(kPeriod, shifted);

        rig.view->updateVisualization(0.5 * kPeriod, kPeriod);
        expectDrawnAtPoseAt(rig, actor, 0.5 * kPeriod, 0.5, kPeriod);
        const double snapped_z = (*userMatrix(actor))(2, 3);

        rig.view->toggleTemporalInterpolation();
        rig.view->updateVisualization(0.5 * kPeriod, kPeriod);
        expectDrawnAtPoseAt(rig, actor, 0.5 * kPeriod, 0.5, kPeriod);
        EXPECT_GT((*userMatrix(actor))(2, 3), snapped_z + 0.05);
    }

    TEST(ContinuumRobotSTL, MeshIsHiddenWithoutAnInterpolator)
    {
        STLFixture fixture;
        Rig rig;
        const auto actor = rig.view->addSTL(fixture.box.string(), 0.5);
        rig.drawOneRow();
        EXPECT_TRUE(rig.backboneVisible());
        EXPECT_FALSE(actor->GetVisibility());
    }

    TEST(ContinuumRobotSTL, MeshIsHiddenWithoutASnapshot)
    {
        STLFixture fixture;
        Rig rig;
        rig.bindInterpolator();
        const auto actor = rig.view->addSTL(fixture.box.string(), 0.5);
        rig.drawOneRow();
        ASSERT_TRUE(actor->GetVisibility());

        rig.view->bindSnapshotSource([](double)
                                      { return std::shared_ptr<const EstimatorSnapshot>(); });
        rig.view->updateVisualization(0.0, 0.0);
        EXPECT_FALSE(actor->GetVisibility());
    }

    TEST(ContinuumRobotSTL, MeshIsHiddenOnceItsRowIsMarginalized)
    {
        STLFixture fixture;
        Rig rig;
        rig.bindInterpolator();
        const auto actor = rig.view->addSTL(fixture.box.string(), 0.5);
        rig.addRow(0.0);
        rig.addRow(kPeriod);
        gtsam::Values window = rig.values;
        for (unsigned int i = 0; i < kSpatialDim; i++)
            for (const gtsam::Key key : rig.state->estimation_nodes.at(i)->keys())
                window.erase(key); // row 0 left the window
        rig.publish(kPeriod, window);

        rig.view->updateVisualization(0.5 * kPeriod, kPeriod);
        EXPECT_FALSE(actor->GetVisibility());
        rig.view->updateVisualization(kPeriod, kPeriod);
        EXPECT_TRUE(actor->GetVisibility());
    }

    TEST(ContinuumRobotSTL, MeshIsHiddenWhenStale)
    {
        STLFixture fixture;
        Rig rig;
        rig.bindInterpolator();
        const auto actor = rig.view->addSTL(fixture.box.string(), 0.5);
        rig.addRow(0.0);
        rig.publish(0.0, rig.values);

        rig.view->updateVisualization(10.0, 10.0); // well past the stale timeout of the only row
        EXPECT_FALSE(rig.backboneVisible());
        EXPECT_FALSE(actor->GetVisibility());
        rig.view->updateVisualization(0.0, 0.0);
        EXPECT_TRUE(actor->GetVisibility());
    }

    TEST(ContinuumRobotSTL, MeshHidesAndShowsWithTheRobot)
    {
        STLFixture fixture;
        Rig rig;
        rig.bindInterpolator();
        const auto actor = rig.view->addSTL(fixture.box.string(), 0.5);
        rig.drawOneRow();
        ASSERT_TRUE(actor->GetVisibility());

        rig.view->toggleVisibility(); // off
        EXPECT_FALSE(actor->GetVisibility()) << "hidden immediately";
        rig.view->updateVisualization(0.0, 0.0);
        EXPECT_FALSE(actor->GetVisibility()) << "and kept hidden by later frames";

        rig.view->toggleVisibility(); // on
        rig.view->updateVisualization(0.0, 0.0);
        EXPECT_TRUE(actor->GetVisibility());
    }

    TEST(ContinuumRobotSTL, MeshIsUnaffectedByTheNodeFramesToggle)
    {
        STLFixture fixture;
        Rig rig;
        rig.bindInterpolator();
        const auto actor = rig.view->addSTL(fixture.box.string(), 0.5);
        rig.drawOneRow();
        rig.view->toggleNodeFrames();
        rig.view->updateVisualization(0.0, 0.0);
        EXPECT_TRUE(actor->GetVisibility());
    }

    TEST(ContinuumRobotSTL, MeshToggleHidesAndShowsTheMeshesImmediately)
    {
        STLFixture fixture;
        Rig rig;
        rig.bindInterpolator();
        const auto first = rig.view->addSTL(fixture.box.string(), 0.25), second = rig.view->addSTL(fixture.box.string(), 0.75);
        rig.drawOneRow();
        ASSERT_TRUE(first->GetVisibility() && second->GetVisibility());

        rig.view->toggleMeshes();
        EXPECT_FALSE(rig.view->meshesShown());
        EXPECT_FALSE(first->GetVisibility() || second->GetVisibility()) << "hidden immediately";
        rig.view->updateVisualization(0.0, 0.0);
        EXPECT_FALSE(first->GetVisibility() || second->GetVisibility()) << "and kept hidden by later frames";
        EXPECT_TRUE(rig.backboneVisible()) << "the backbone is unaffected";

        rig.view->toggleMeshes();
        EXPECT_TRUE(rig.view->meshesShown());
        EXPECT_TRUE(first->GetVisibility() && second->GetVisibility()) << "shown immediately, where they were last drawn";
        expectDrawnAtPoseAt(rig, first, 0.0, 0.25, 0.0);
        rig.view->updateVisualization(0.0, 0.0);
        EXPECT_TRUE(first->GetVisibility() && second->GetVisibility());
    }

    TEST(ContinuumRobotSTL, MeshesCanStartHidden)
    {
        STLFixture fixture;
        Rig rig;
        rig.bindInterpolator();
        rig.view->show_meshes = false;
        const auto actor = rig.view->addSTL(fixture.box.string(), 0.5);
        rig.drawOneRow();
        EXPECT_TRUE(rig.backboneVisible());
        EXPECT_FALSE(actor->GetVisibility());
    }

    // Toggling the meshes back on never shows one that the robot's own state hides.
    TEST(ContinuumRobotSTL, MeshToggleDoesNotShowMeshesTheRobotHides)
    {
        STLFixture fixture;
        Rig rig;
        rig.bindInterpolator();
        const auto actor = rig.view->addSTL(fixture.box.string(), 0.5);
        rig.drawOneRow();

        rig.view->toggleMeshes();     // off
        rig.view->toggleVisibility(); // robot off
        rig.view->toggleMeshes();     // on
        EXPECT_FALSE(actor->GetVisibility()) << "the robot is hidden";
        rig.view->toggleVisibility(); // robot on
        rig.view->updateVisualization(0.0, 0.0);
        EXPECT_TRUE(actor->GetVisibility());

        // Never drawn (no snapshot yet): nothing to show.
        Rig fresh;
        fresh.bindInterpolator();
        const auto undrawn = fresh.view->addSTL(fixture.box.string(), 0.5);
        fresh.view->toggleMeshes();
        fresh.view->toggleMeshes();
        EXPECT_FALSE(undrawn->GetVisibility());
    }

    TEST(ContinuumRobotSTL, ShowingTheRobotAgainKeepsHiddenMeshesHidden)
    {
        STLFixture fixture;
        Rig rig;
        rig.bindInterpolator();
        const auto actor = rig.view->addSTL(fixture.box.string(), 0.5);
        rig.drawOneRow();
        rig.view->toggleMeshes();
        rig.view->toggleVisibility(); // off
        rig.view->toggleVisibility(); // on
        rig.view->updateVisualization(0.0, 0.0);
        EXPECT_TRUE(rig.backboneVisible());
        EXPECT_FALSE(actor->GetVisibility());
    }

    // A mesh added before the view is visualized joins the renderer then; one added after joins it at once.
    TEST(ContinuumRobotSTL, MeshesAreAddedToTheRendererWhetherAddedBeforeOrAfterIt)
    {
        STLFixture fixture;
        Rig rig;
        vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
        const auto before = rig.view->addSTL(fixture.box.string(), 0.25);
        EXPECT_EQ(renderer->GetActors()->GetNumberOfItems(), 0);

        rig.view->initMeshes(renderer);
        EXPECT_EQ(renderer->GetActors()->GetNumberOfItems(), 1);
        EXPECT_TRUE(renderer->GetActors()->IsItemPresent(before));

        const auto after = rig.view->addSTL(fixture.box.string(), 0.75);
        EXPECT_EQ(renderer->GetActors()->GetNumberOfItems(), 2);
        EXPECT_TRUE(renderer->GetActors()->IsItemPresent(after));
    }
} // namespace Spacetime
