#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCylinderSource.h>
#include <vtkDataArray.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkSphereSource.h>

#include <spacetime/visualizer/default/utilities.hpp>

namespace Spacetime::Visualization::Default::Utilities
{
    namespace
    {
        Eigen::Matrix4d toEigen(const vtkSmartPointer<vtkMatrix4x4> &matrix)
        {
            Eigen::Matrix4d out;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    out(r, c) = matrix->GetElement(r, c);
            return out;
        }

        Eigen::Matrix4d toEigen(vtkMatrix4x4 *matrix)
        {
            Eigen::Matrix4d out;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    out(r, c) = matrix->GetElement(r, c);
            return out;
        }

        Eigen::Matrix4d somePose(double angle, const Eigen::Vector3d &position)
        {
            Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
            pose.block<3, 3>(0, 0) = Eigen::AngleAxisd(angle, Eigen::Vector3d(1, -2, 3).normalized()).toRotationMatrix();
            pose.block<3, 1>(0, 3) = position;
            return pose;
        }

        /// The prop's user matrix is created once and then updated in place, and its computed matrix follows it.
        void expectUserMatrixReused(vtkProp3D *prop)
        {
            const Eigen::Matrix4d first = somePose(0.3, Eigen::Vector3d(0.1, 0.2, 0.3)), second = somePose(-0.8, Eigen::Vector3d(-1, 0.5, 2));
            setUserMatrix(prop, first);
            vtkMatrix4x4 *matrix = prop->GetUserMatrix();
            ASSERT_NE(matrix, nullptr);
            EXPECT_TRUE(toEigen(matrix).isApprox(first));
            EXPECT_TRUE(toEigen(prop->GetMatrix()).isApprox(first));

            setUserMatrix(prop, second);
            EXPECT_EQ(prop->GetUserMatrix(), matrix) << "the matrix must be reused, not reallocated";
            EXPECT_TRUE(toEigen(matrix).isApprox(second));
            EXPECT_TRUE(toEigen(prop->GetMatrix()).isApprox(second)) << "the prop must see the in-place update";
        }

        /// The transform's linear part L maps the unit sphere onto the ellipsoid
        /// {x : x^T (scale^2 Sigma)^-1 x = 1} exactly when L L^T = scale^2 Sigma.
        void expectEllipsoidOf(const Eigen::Matrix4d &transform, const Eigen::Matrix3d &covariance, double scale)
        {
            const Eigen::Matrix3d L = transform.block<3, 3>(0, 0);
            EXPECT_TRUE((L * L.transpose()).isApprox(scale * scale * covariance, 1e-9));
            EXPECT_GT(L.determinant(), 0.0) << "the transform must not mirror the sphere";
            EXPECT_TRUE(transform.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1)));
        }
    } // namespace

    TEST(VisualizerUtilities, EllipsoidScaleIsThe997PercentChiSquaredRadius)
    {
        EXPECT_NEAR(kEllipsoidScale99_7 * kEllipsoidScale99_7, 13.9314, 1e-3);
    }

    TEST(VisualizerUtilities, EllipsoidOfADiagonalCovariance)
    {
        const Eigen::Vector3d position(0.1, -0.2, 0.3);
        const Eigen::Matrix3d covariance = Eigen::Vector3d(4.0, 1.0, 9.0).asDiagonal();
        const Eigen::Matrix4d transform = toEigen(getEllipsoidMatrix(position, covariance, 1.0));

        expectEllipsoidOf(transform, covariance, 1.0);
        EXPECT_TRUE(transform.col(3).head(3).isApprox(position));
        // The semi-axes are the standard deviations along each eigenvector.
        Eigen::Vector3d lengths;
        for (int i = 0; i < 3; i++)
            lengths(i) = transform.block<3, 1>(0, i).norm();
        std::sort(lengths.data(), lengths.data() + 3);
        EXPECT_TRUE(lengths.isApprox(Eigen::Vector3d(1.0, 2.0, 3.0)));
    }

    TEST(VisualizerUtilities, EllipsoidOfARotatedCovarianceUsesTheDefaultScale)
    {
        const Eigen::Matrix3d R = Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, -1).normalized()).toRotationMatrix();
        const Eigen::Matrix3d covariance = R * Eigen::Vector3d(1e-4, 4e-4, 2.5e-5).asDiagonal() * R.transpose();
        const Eigen::Matrix4d transform = toEigen(getEllipsoidMatrix(Eigen::Vector3d::Zero(), covariance));
        expectEllipsoidOf(transform, covariance, kEllipsoidScale99_7);
    }

    TEST(VisualizerUtilities, EllipsoidOfADegenerateCovarianceStaysFinite)
    {
        // Zero, and very slightly indefinite (as numerical posteriors can be): clamped,
        // never NaN, and never collapsed to an exactly singular transform.
        const Eigen::Matrix3d zero = Eigen::Matrix3d::Zero();
        const Eigen::Matrix3d indefinite = Eigen::Vector3d(1e-6, -1e-15, 4e-6).asDiagonal();
        for (const Eigen::Matrix3d &covariance : {zero, indefinite})
        {
            const Eigen::Matrix4d transform = toEigen(getEllipsoidMatrix(Eigen::Vector3d(1, 2, 3), covariance));
            EXPECT_TRUE(transform.allFinite());
            EXPECT_GT(transform.topLeftCorner(3, 3).determinant(), 0.0);
            for (int i = 0; i < 3; i++)
                EXPECT_GE(transform.col(i).head(3).norm(), 1e-8 - 1e-20);
        }
    }

    TEST(VisualizerUtilities, EllipsoidSymmetrizesItsInput)
    {
        Eigen::Matrix3d covariance = Eigen::Vector3d(1.0, 2.0, 3.0).asDiagonal();
        Eigen::Matrix3d asymmetric = covariance;
        asymmetric(0, 1) += 1e-3;
        asymmetric(1, 0) -= 1e-3;
        const Eigen::Matrix4d transform = toEigen(getEllipsoidMatrix(Eigen::Vector3d::Zero(), asymmetric, 1.0));
        expectEllipsoidOf(transform, covariance, 1.0);
    }

    TEST(VisualizerUtilities, EllipsoidTransformMatchesTheEllipsoidMatrix)
    {
        const Eigen::Vector3d position(0.4, -0.1, 0.2);
        const Eigen::Matrix3d covariance = Eigen::Vector3d(2.0, 0.5, 1.0).asDiagonal();
        EXPECT_TRUE(ellipsoidTransform(position, covariance).isApprox(toEigen(getEllipsoidMatrix(position, covariance))));
        EXPECT_TRUE(ellipsoidTransform(position, covariance, 1.0).isApprox(toEigen(getEllipsoidMatrix(position, covariance, 1.0))));
    }

    TEST(VisualizerUtilities, FrameMeshSpansEachAxisToItsLength)
    {
        constexpr double kLength = 0.2;
        const auto mapper = createFrameMapper(0.05, kLength);
        mapper->Update();
        double bounds[6];
        mapper->GetInput()->GetBounds(bounds);
        constexpr double kFloatTolerance = 1e-7 * kLength; // the sources' points are single precision
        for (int i = 0; i < 3; i++)
        {
            EXPECT_NEAR(bounds[2 * i + 1], kLength, kFloatTolerance) << "axis " << i << " ends at its tip";
            EXPECT_NEAR(bounds[2 * i], -0.08 * kLength, kFloatTolerance) << "axis " << i << ": the other axes' cones reach back by their radius";
        }
    }

    TEST(VisualizerUtilities, FrameMeshIsColouredRedGreenBlueByAxis)
    {
        constexpr double kLength = 0.2;
        const auto mapper = createFrameMapper(0.05, kLength);
        mapper->Update();
        vtkPolyData *mesh = mapper->GetInput();
        vtkDataArray *colors = mesh->GetPointData()->GetScalars();
        ASSERT_NE(colors, nullptr);
        ASSERT_EQ(colors->GetNumberOfComponents(), 3);
        EXPECT_TRUE(mapper->GetScalarVisibility());
        EXPECT_EQ(mapper->GetScalarMode(), VTK_SCALAR_MODE_USE_POINT_DATA);
        EXPECT_EQ(mapper->GetColorMode(), VTK_COLOR_MODE_DIRECT_SCALARS);

        // Beyond 0.1 L (wider than every shaft and cone), a point lies on exactly one axis.
        int counted[3] = {0, 0, 0};
        for (vtkIdType id = 0; id < mesh->GetNumberOfPoints(); id++)
        {
            double point[3], rgb[3];
            mesh->GetPoint(id, point);
            colors->GetTuple(id, rgb);
            for (int axis = 0; axis < 3; axis++)
                if (point[axis] > 0.1 * kLength)
                {
                    counted[axis]++;
                    for (int channel = 0; channel < 3; channel++)
                        EXPECT_DOUBLE_EQ(rgb[channel], channel == axis ? 255.0 : 0.0) << "point " << id << " on axis " << axis;
                }
        }
        for (int axis = 0; axis < 3; axis++)
            EXPECT_GT(counted[axis], 0) << axis;
    }

    TEST(VisualizerUtilities, FrameActorsShareAGivenMapper)
    {
        const auto mapper = createFrameMapper(0.05, 0.1);
        const auto a = createFrameActor(mapper), b = createFrameActor(mapper);
        EXPECT_EQ(a->GetMapper(), mapper.Get());
        EXPECT_EQ(b->GetMapper(), mapper.Get());
        EXPECT_TRUE(a->GetVisibility());

        const auto own = createFrameActor(0.05, 0.1);
        ASSERT_NE(own->GetMapper(), nullptr);
        EXPECT_NE(own->GetMapper(), mapper.Get());
        EXPECT_TRUE(own->GetVisibility());
    }

    TEST(VisualizerUtilities, UnitSphereMapperDrawsAUnitSphereAtTheSharedResolution)
    {
        const auto mapper = createUnitSphereMapper();
        auto *sphere = vtkSphereSource::SafeDownCast(mapper->GetInputAlgorithm());
        ASSERT_NE(sphere, nullptr);
        EXPECT_DOUBLE_EQ(sphere->GetRadius(), 1.0);
        EXPECT_EQ(sphere->GetThetaResolution(), kSphereResolution);
        EXPECT_EQ(sphere->GetPhiResolution(), kSphereResolution);
    }

    TEST(VisualizerUtilities, UnitCylinderMapperDrawsAUnitCylinderAtTheSharedResolution)
    {
        const auto mapper = createUnitCylinderMapper();
        auto *cylinder = vtkCylinderSource::SafeDownCast(mapper->GetInputAlgorithm());
        ASSERT_NE(cylinder, nullptr);
        EXPECT_DOUBLE_EQ(cylinder->GetRadius(), 1.0);
        EXPECT_DOUBLE_EQ(cylinder->GetHeight(), 1.0); // see cylinderMatrixBetween
        EXPECT_EQ(cylinder->GetResolution(), kCylinderResolution);
    }

    TEST(VisualizerUtilities, SetUserMatrixUpdatesAnActorsMatrixInPlace)
    {
        vtkNew<vtkActor> actor;
        expectUserMatrixReused(actor);
    }

    TEST(VisualizerUtilities, SetUserMatrixUpdatesAFrameActorsMatrixInPlace)
    {
        const auto frame = createFrameActor(0.01, 0.1);
        expectUserMatrixReused(frame);
    }

    TEST(VisualizerUtilities, SetUserMatrixWithAnUnchangedPoseDoesNotModifyTheMatrix)
    {
        vtkNew<vtkActor> actor;
        const Eigen::Matrix4d pose = somePose(0.5, Eigen::Vector3d(1, 2, 3));
        setUserMatrix(actor, pose);
        const vtkMTimeType before = actor->GetUserMatrix()->GetMTime();
        setUserMatrix(actor, pose);
        EXPECT_EQ(actor->GetUserMatrix()->GetMTime(), before) << "an unchanged pose must not trigger a re-render of the prop";
    }

    TEST(VisualizerUtilities, SurfaceMaterialHasAVisibleHighlight)
    {
        vtkNew<vtkProperty> property;
        applySurfaceMaterial(property);
        EXPECT_DOUBLE_EQ(property->GetAmbient(), kSurfaceAmbient);
        EXPECT_DOUBLE_EQ(property->GetDiffuse(), kSurfaceDiffuse);
        EXPECT_DOUBLE_EQ(property->GetSpecular(), kSurfaceSpecular);
        EXPECT_DOUBLE_EQ(property->GetSpecularPower(), kSurfaceSpecularPower);
        EXPECT_GE(kSurfaceSpecular, 0.3) << "dark surfaces show their shape mostly through the highlight";
    }

    TEST(VisualizerUtilities, SurfaceShadesAreLightEnoughToShade)
    {
        // Lit surfaces only show their shape if their color is far enough from black.
        for (unsigned char shade : kBackboneShades)
            EXPECT_GE(shade, 100);
        EXPECT_NE(kBackboneShades[0], kBackboneShades[1]) << "neighbouring segments stay distinguishable";
        EXPECT_GE(kJointGrey, 0.2);
        EXPECT_LT(kJointGrey * 255.0, kBackboneShades[0]) << "joints stay darker than the backbones they join";
    }

    namespace
    {
        Eigen::Vector3d position(vtkCamera *camera) { return Eigen::Vector3d(camera->GetPosition()); }
        Eigen::Vector3d focalPoint(vtkCamera *camera) { return Eigen::Vector3d(camera->GetFocalPoint()); }
        Eigen::Vector3d viewUp(vtkCamera *camera) { return Eigen::Vector3d(camera->GetViewUp()); }
    } // namespace

    TEST(VisualizerUtilities, OrbitTurnsTheCameraAboutTheVerticalThroughItsFocalPoint)
    {
        vtkNew<vtkCamera> camera;
        camera->SetFocalPoint(0.5, -0.2, 0.3);
        camera->SetPosition(1.5, -0.2, 0.8); // 1 m along +x of the focal point, 0.5 m above it
        camera->SetViewUp(0.0, 0.0, 1.0);

        orbitAboutVertical(camera, M_PI / 2.0);
        EXPECT_TRUE(focalPoint(camera).isApprox(Eigen::Vector3d(0.5, -0.2, 0.3), 1e-12)) << "it keeps looking at the same point";
        EXPECT_TRUE(position(camera).isApprox(Eigen::Vector3d(0.5, 0.8, 0.8), 1e-12)) << "a quarter turn, counter-clockwise from above";
        EXPECT_TRUE(viewUp(camera).isApprox(Eigen::Vector3d::UnitZ(), 1e-12));
    }

    TEST(VisualizerUtilities, OrbitKeepsHeightAndDistanceAndTurnsATiltedViewUp)
    {
        vtkNew<vtkCamera> camera;
        camera->SetFocalPoint(0.0, 0.0, 0.0);
        camera->SetPosition(0.6, 0.8, 1.0);
        camera->SetViewUp(-0.3, -0.4, 0.5); // perpendicular to the view direction, tilted
        const Eigen::Vector3d up_before = viewUp(camera);
        const double distance = camera->GetDistance();

        const double angle = 0.7;
        orbitAboutVertical(camera, angle);
        EXPECT_NEAR(position(camera).z(), 1.0, 1e-12);
        EXPECT_NEAR(camera->GetDistance(), distance, 1e-12);
        EXPECT_TRUE(viewUp(camera).isApprox(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()) * up_before, 1e-9)) << "the view turns with the camera";

        orbitAboutVertical(camera, -angle);
        EXPECT_TRUE(position(camera).isApprox(Eigen::Vector3d(0.6, 0.8, 1.0), 1e-12)) << "and turns back";
    }

    TEST(VisualizerUtilities, OrbitByZeroLeavesTheCameraUnmodified)
    {
        vtkNew<vtkCamera> camera;
        camera->SetPosition(1.0, 2.0, 3.0);
        const vtkMTimeType before = camera->GetMTime();
        orbitAboutVertical(camera, 0.0);
        EXPECT_EQ(camera->GetMTime(), before);
    }
} // namespace Spacetime::Visualization::Default::Utilities
