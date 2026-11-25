#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <vtkActor.h>
#include <vtkPlaneSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>

#include <spacetime/visualizer/default/objects/GroundPlane.hpp>

namespace Spacetime
{
    namespace
    {
        using Visualization::GroundPlaneOptions;
        using Visualization::Default::GroundPlane;

        /// The corners span a square of side `size` centred at `position`, in the plane normal to `normal`.
        void expectSquare(const GroundPlane &plane, const Eigen::Vector3d &normal, const Eigen::Vector3d &position, double size)
        {
            const auto c = plane.corners();
            const Eigen::Vector3d n = normal.normalized();
            const Eigen::Vector3d edge1 = c.point1 - c.origin, edge2 = c.point2 - c.origin;
            EXPECT_NEAR(edge1.norm(), size, 1e-12);
            EXPECT_NEAR(edge2.norm(), size, 1e-12);
            EXPECT_NEAR(edge1.dot(edge2), 0.0, 1e-12);
            EXPECT_NEAR(edge1.dot(n), 0.0, 1e-12);
            EXPECT_NEAR(edge2.dot(n), 0.0, 1e-12);
            EXPECT_TRUE(edge1.cross(edge2).normalized().isApprox(n, 1e-12)); // right-handed about the normal
            EXPECT_TRUE((0.5 * (c.origin + c.opposite)).isApprox(position, 1e-12));
            EXPECT_TRUE(c.opposite.isApprox(c.point1 + c.point2 - c.origin, 1e-12));
            EXPECT_TRUE(plane.options().normal.isApprox(n, 1e-12));
        }
    } // namespace

    TEST(GroundPlaneVisualization, DefaultIsAUnitSquareAtTheOrigin)
    {
        const GroundPlane plane;
        expectSquare(plane, Eigen::Vector3d::UnitZ(), Eigen::Vector3d::Zero(), 1.0);
        // Axis-aligned: edges along world x and y.
        const auto c = plane.corners();
        EXPECT_TRUE(c.origin.isApprox(Eigen::Vector3d(-0.5, -0.5, 0), 1e-12));
        EXPECT_TRUE(c.point1.isApprox(Eigen::Vector3d(0.5, -0.5, 0), 1e-12));
        EXPECT_TRUE(c.point2.isApprox(Eigen::Vector3d(-0.5, 0.5, 0), 1e-12));
    }

    TEST(GroundPlaneVisualization, TiltedPlaneIsCentredAndNormal)
    {
        const Eigen::Vector3d normal(0.3, -1.0, 2.0), position(0.1, 0.2, -0.3);
        expectSquare(GroundPlane(normal, position, 0.4), normal, position, 0.4);
    }

    TEST(GroundPlaneVisualization, NormalAlongWorldXStillHasASquare)
    {
        const Eigen::Vector3d normal(-2.0, 0.0, 0.0), position(1.0, 0.0, 0.5);
        expectSquare(GroundPlane(GroundPlaneOptions{normal, position, 0.25}), normal, position, 0.25);
    }

    TEST(GroundPlaneVisualization, InvalidOptionsThrow)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        EXPECT_THROW(GroundPlane(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 1.0), std::invalid_argument);
        EXPECT_THROW(GroundPlane(Eigen::Vector3d(nan, 0, 1), Eigen::Vector3d::Zero(), 1.0), std::invalid_argument);
        EXPECT_THROW(GroundPlane(Eigen::Vector3d::UnitZ(), Eigen::Vector3d(nan, 0, 0), 1.0), std::invalid_argument);
        for (double size : {0.0, -1.0, nan, std::numeric_limits<double>::infinity()})
            EXPECT_THROW(GroundPlane(Eigen::Vector3d::UnitZ(), Eigen::Vector3d::Zero(), size), std::invalid_argument) << size;
    }

    TEST(GroundPlaneVisualization, ActorIsThePlane)
    {
        const GroundPlane plane(Eigen::Vector3d::UnitY(), Eigen::Vector3d(0, 1, 0), 2.0);
        EXPECT_EQ(plane.actor(), nullptr);
        vtkActor *actor = plane.createActor();
        ASSERT_NE(actor, nullptr);
        EXPECT_EQ(plane.createActor(), actor); // created once
        EXPECT_EQ(plane.actor(), actor);
        EXPECT_TRUE(actor->GetVisibility());
        auto *source = vtkPlaneSource::SafeDownCast(vtkPolyDataMapper::SafeDownCast(actor->GetMapper())->GetInputAlgorithm());
        ASSERT_NE(source, nullptr);
        double center[3], normal[3];
        source->GetCenter(center);
        source->GetNormal(normal);
        EXPECT_TRUE(Eigen::Vector3d(center[0], center[1], center[2]).isApprox(Eigen::Vector3d(0, 1, 0), 1e-12));
        EXPECT_TRUE(Eigen::Vector3d(normal[0], normal[1], normal[2]).isApprox(Eigen::Vector3d::UnitY(), 1e-12));
    }

    TEST(GroundPlaneVisualization, ToggleFlipsTheActor)
    {
        const GroundPlane plane;
        vtkActor *actor = plane.createActor();
        plane.toggleVisibility();
        EXPECT_FALSE(actor->GetVisibility());
        plane.toggleVisibility();
        EXPECT_TRUE(actor->GetVisibility());
    }

    TEST(GroundPlaneVisualization, StartsHiddenWhenSetInvisible)
    {
        GroundPlane plane;
        plane.setVisible(false);
        EXPECT_FALSE(plane.createActor()->GetVisibility());
        plane.toggleVisibility();
        EXPECT_TRUE(plane.actor()->GetVisibility());
    }
} // namespace Spacetime
