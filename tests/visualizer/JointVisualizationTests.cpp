#include <memory>
#include <optional>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>
#include <vtkActor.h>
#include <vtkMatrix4x4.h>
#include <vtkProperty.h>

#include <spacetime/visualizer/default/objects/Joint.hpp>
#include <spacetime/visualizer/default/utilities.hpp>

namespace Spacetime
{
    namespace
    {
        using Visualization::Default::Joint;

        Eigen::Matrix4d toEigen(vtkMatrix4x4 *matrix)
        {
            Eigen::Matrix4d out;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    out(r, c) = matrix->GetElement(r, c);
            return out;
        }

        Eigen::Vector3d apply(const Eigen::Matrix4d &T, const Eigen::Vector3d &p) { return (T * p.homogeneous()).head<3>(); }

        Joint::PointPoseFn fixedPose(const Eigen::Vector3d &position)
        {
            const Eigen::Matrix4d pose = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.3, -0.2, 0.1), gtsam::Point3(position)).matrix();
            return [pose](double, double) { return std::optional<Eigen::Matrix4d>(pose); };
        }

        /// Dark grey (not black, which no light can brighten), with the shared surface material.
        void expectJointMaterial(const Joint &joint)
        {
            vtkProperty *property = joint.mp_actor->GetProperty();
            double rgb[3];
            property->GetColor(rgb);
            for (double channel : rgb)
                EXPECT_DOUBLE_EQ(channel, Visualization::Default::Utilities::kJointGrey);
            EXPECT_DOUBLE_EQ(property->GetSpecular(), Visualization::Default::Utilities::kSurfaceSpecular);
        }
    } // namespace

    TEST(JointVisualization, CylinderMatrixSpansTheSegment)
    {
        const Eigen::Vector3d p0(0.1, -0.2, 0.3), p1(0.4, 0.1, -0.1);
        const Eigen::Matrix4d T = Visualization::Default::Utilities::cylinderMatrixBetween(p0, p1, 0.02);
        // The unit cylinder runs along its local y from -0.5 to 0.5, with radius 1.
        EXPECT_TRUE(apply(T, Eigen::Vector3d(0, -0.5, 0)).isApprox(p0, 1e-12));
        EXPECT_TRUE(apply(T, Eigen::Vector3d(0, 0.5, 0)).isApprox(p1, 1e-12));
        const Eigen::Vector3d axis = (p1 - p0).normalized();
        const Eigen::Vector3d rim = T.block<3, 3>(0, 0) * Eigen::Vector3d::UnitX();
        EXPECT_NEAR(rim.norm(), 0.02, 1e-12);
        EXPECT_NEAR(rim.dot(axis), 0.0, 1e-12);
    }

    TEST(JointVisualization, CylinderMatrixHandlesAntiparallelAndDegenerateSegments)
    {
        const Eigen::Vector3d p0(0, 0, 0), down(0, -0.3, 0);
        const Eigen::Matrix4d T = Visualization::Default::Utilities::cylinderMatrixBetween(p0, down, 0.01);
        EXPECT_TRUE(apply(T, Eigen::Vector3d(0, 0.5, 0)).isApprox(down, 1e-12));
        EXPECT_TRUE(T.allFinite());
        EXPECT_TRUE(Visualization::Default::Utilities::cylinderMatrixBetween(p0, p0, 0.01).allFinite());
    }

    TEST(JointVisualization, OffsetJointIsACylinderBetweenThePoints)
    {
        const Eigen::Vector3d p_a(0.1, 0.0, 0.4), p_b(0.15, 0.02, 0.38);
        Joint joint(fixedPose(p_a), fixedPose(p_b), true);
        joint.radius = 0.012;
        joint.initActors();
        joint.updateVisualization(1.0, 1.0);

        ASSERT_TRUE(joint.mp_actor);
        EXPECT_TRUE(joint.mp_actor->GetVisibility());
        expectJointMaterial(joint);
        EXPECT_TRUE(joint.drawsLink());
        const Eigen::Matrix4d T = toEigen(joint.mp_actor->GetUserMatrix());
        EXPECT_TRUE(apply(T, Eigen::Vector3d(0, -0.5, 0)).isApprox(p_a, 1e-12));
        EXPECT_TRUE(apply(T, Eigen::Vector3d(0, 0.5, 0)).isApprox(p_b, 1e-12));
        EXPECT_NEAR((T.block<3, 3>(0, 0) * Eigen::Vector3d::UnitY()).norm(), (p_b - p_a).norm(), 1e-12);
    }

    TEST(JointVisualization, JointWithoutOffsetIsASphere)
    {
        const Eigen::Vector3d p_a(0.1, 0.0, 0.4), p_b(0.1002, 0.0, 0.4); // a nearly-satisfied joint
        Joint joint(fixedPose(p_a), fixedPose(p_b), false);
        joint.radius = 0.012;
        joint.initActors();
        joint.updateVisualization(1.0, 1.0);

        EXPECT_FALSE(joint.drawsLink());
        EXPECT_TRUE(joint.mp_actor->GetVisibility());
        expectJointMaterial(joint);
        const Eigen::Matrix4d T = toEigen(joint.mp_actor->GetUserMatrix());
        EXPECT_TRUE((T.block<3, 1>(0, 3).isApprox(0.5 * (p_a + p_b), 1e-12)));
        EXPECT_TRUE((T.block<3, 3>(0, 0).isApprox(Eigen::Matrix3d::Identity() * 0.012, 1e-12))); // unit sphere scaled to the radius
    }

    TEST(JointVisualization, HiddenWhenEitherPointIsUnavailable)
    {
        const Joint::PointPoseFn missing = [](double, double) { return std::optional<Eigen::Matrix4d>(); };
        for (bool has_offset : {false, true})
        {
            Joint missing_a(missing, fixedPose(Eigen::Vector3d::Zero()), has_offset);
            missing_a.initActors();
            missing_a.updateVisualization(1.0, 1.0);
            EXPECT_FALSE(missing_a.mp_actor->GetVisibility());

            Joint missing_b(fixedPose(Eigen::Vector3d::Zero()), missing, has_offset);
            missing_b.initActors();
            missing_b.updateVisualization(1.0, 1.0);
            EXPECT_FALSE(missing_b.mp_actor->GetVisibility());
        }
    }

    TEST(JointVisualization, ToggledOffJointStaysHidden)
    {
        Joint joint(fixedPose(Eigen::Vector3d::Zero()), fixedPose(Eigen::Vector3d(0.1, 0, 0)), true);
        joint.initActors();
        joint.toggleVisibility();
        joint.updateVisualization(1.0, 1.0);
        EXPECT_FALSE(joint.mp_actor->GetVisibility());
    }

    TEST(JointVisualization, NullPoseSourcesThrow)
    {
        EXPECT_THROW(Joint(nullptr, fixedPose(Eigen::Vector3d::Zero()), true), std::invalid_argument);
        EXPECT_THROW(Joint(fixedPose(Eigen::Vector3d::Zero()), nullptr, true), std::invalid_argument);
    }

    TEST(JointVisualization, JointsGivenOneMapperShareIt)
    {
        for (bool has_offset : {false, true})
        {
            const auto mapper = has_offset ? Visualization::Default::Utilities::createUnitCylinderMapper()
                                           : Visualization::Default::Utilities::createUnitSphereMapper();
            Joint a(fixedPose(Eigen::Vector3d::Zero()), fixedPose(Eigen::Vector3d(0.1, 0, 0)), has_offset);
            Joint b(fixedPose(Eigen::Vector3d::Zero()), fixedPose(Eigen::Vector3d(0, 0.1, 0)), has_offset);
            a.initActors(mapper);
            b.initActors(mapper);
            EXPECT_EQ(a.mp_actor->GetMapper(), mapper.Get()) << has_offset;
            EXPECT_EQ(b.mp_actor->GetMapper(), mapper.Get()) << has_offset;
        }
    }

    TEST(JointVisualization, MatrixIsUpdatedInPlaceAsThePointsMove)
    {
        Eigen::Vector3d p_b(0.1, 0.0, 0.0);
        const Joint::PointPoseFn moving = [&p_b](double, double)
        {
            Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
            pose.block<3, 1>(0, 3) = p_b;
            return std::optional<Eigen::Matrix4d>(pose);
        };
        Joint joint(fixedPose(Eigen::Vector3d::Zero()), moving, false);
        joint.initActors();
        joint.updateVisualization(1.0, 1.0);
        vtkMatrix4x4 *matrix = joint.mp_actor->GetUserMatrix();
        ASSERT_NE(matrix, nullptr);

        p_b = Eigen::Vector3d(0.3, 0.1, 0.0);
        joint.updateVisualization(2.0, 2.0);
        EXPECT_EQ(joint.mp_actor->GetUserMatrix(), matrix);
        EXPECT_TRUE((toEigen(matrix).block<3, 1>(0, 3).isApprox(0.5 * p_b, 1e-12)));
    }
} // namespace Spacetime
