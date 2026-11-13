#ifndef DEFAULT_VISUALIZATIONUTILITIES_HPP
#define DEFAULT_VISUALIZATIONUTILITIES_HPP

#include <array>
#include <string>

#include <Eigen/Core>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkProp3D.h>
#include <vtkProperty.h>

#include <spacetime/estimator/utilities/Ellipsoid.hpp>

namespace Spacetime::Visualization::Default::Utilities
{
    /// Theta and phi resolution of createUnitSphereMapper's sphere.
    constexpr int kSphereResolution = 16;
    /// Resolution of createUnitCylinderMapper's cylinder.
    constexpr int kCylinderResolution = 16;
    /// Sides of a robot's backbone tube.
    constexpr int kTubeSides = 16;
    /// Resolution of a frame's shafts and tips (see createFrameMapper).
    constexpr int kFrameResolution = 16;

    /// Grey levels (0-255) a robot's backbone alternates between, one per segment between nodes.
    /// Light enough for the scene's lights to shade (black shows no shape).
    constexpr std::array<unsigned char, 2> kBackboneShades{110, 150};
    /// Grey level (0-1) of a joint: darker than the backbones it joins, but still shaded.
    constexpr double kJointGrey = 0.25;
    /// The shared surface material of backbones and joints (see applySurfaceMaterial).
    constexpr double kSurfaceAmbient = 0.3, kSurfaceDiffuse = 0.6, kSurfaceSpecular = 0.4, kSurfaceSpecularPower = 20.0;

    /// Gives `property` the shared surface material: kSurfaceAmbient, kSurfaceDiffuse, and a visible kSurfaceSpecular highlight.
    void applySurfaceMaterial(vtkProperty *property);

    /**
     * @brief Mapper for an XYZ frame (x red, y green, z blue) as one mesh, so a frame is a
     *        single actor (a vtkAxesActor is six, and rebuilds its parts on every render).
     *        Each axis is `totalLength` long, as vtkAxesActor draws it: a cylindrical shaft over
     *        its first 80% (radius `cylinderRadius` relative to the shaft length) and a cone tip
     *        over the rest (radius 40% of the tip length). To be shared by frames of one size.
     */
    vtkSmartPointer<vtkPolyDataMapper> createFrameMapper(double cylinderRadius, double totalLength);

    /// Visible frame actor drawing `mapper` (see createFrameMapper).
    vtkSmartPointer<vtkActor> createFrameActor(vtkSmartPointer<vtkPolyDataMapper> mapper);

    /// Visible frame actor with a mapper of its own (see createFrameMapper).
    vtkSmartPointer<vtkActor> createFrameActor(double cylinderRadius, double totalLength);

    /// Translucent (opacity 0.15) ellipsoid actor drawing `mapper`, blue or (if `red`) red. Created hidden.
    vtkSmartPointer<vtkActor> createEllipsoidActor(vtkSmartPointer<vtkPolyDataMapper> mapper, bool red = false);

    /// Mapper for a unit sphere, to be shared by ellipsoid actors (see createEllipsoidActor, getEllipsoidMatrix).
    vtkSmartPointer<vtkPolyDataMapper> createUnitSphereMapper();

    /// Mapper for a unit cylinder (see cylinderMatrixBetween), to be shared by joint links.
    vtkSmartPointer<vtkPolyDataMapper> createUnitCylinderMapper();

    // The confidence ellipsoid (see Spacetime::Utilities::ellipsoidTransform), shared with other backends.
    using ::Spacetime::Utilities::ellipsoidTransform;
    using ::Spacetime::Utilities::kEllipsoidScale99_7;

    /// ellipsoidTransform as a new vtkMatrix4x4.
    vtkSmartPointer<vtkMatrix4x4> getEllipsoidMatrix(const Eigen::Vector3d &position, const Eigen::Matrix3d &covariance, double scale = kEllipsoidScale99_7);

    /**
     * @brief Transform mapping a unit cylinder (radius 1, along its local y from -0.5 to
     *        0.5, as vtkCylinderSource builds it with SetRadius(1) and SetHeight(1)) onto
     *        the segment from `p0` to `p1` with the given `radius`. A degenerate (zero-length)
     *        segment gives a flat disc at `p0`.
     */
    Eigen::Matrix4d cylinderMatrixBetween(const Eigen::Vector3d &p0, const Eigen::Vector3d &p1, double radius);

    /**
     * @brief Reads the STL mesh at `path` (e.g. an environment scan or a part of a robot), uniformly
     *        scaled by `scale` (e.g. 0.001 for a mesh in millimetres).
     * @throws std::runtime_error if the file is missing or holds no geometry.
     */
    vtkSmartPointer<vtkPolyData> loadSTLMesh(const std::string &path, double scale = 1.0);

    /// Copies `matrix` into a new vtkMatrix4x4.
    vtkSmartPointer<vtkMatrix4x4> toVtkMatrix(const Eigen::Matrix4d &matrix);

    /// Turns `camera` by `angle` [rad] about the world vertical (z) through its focal point, counter-clockwise
    /// seen from above: a turntable orbit keeping its height, distance and focal point. A zero angle is a no-op.
    void orbitAboutVertical(vtkCamera *camera, double angle);

    /// Sets `prop`'s user matrix to `matrix`, in place once it has one (no allocation per frame,
    /// and an unchanged matrix leaves the prop unmodified).
    void setUserMatrix(vtkProp3D *prop, const Eigen::Matrix4d &matrix);
}

#endif // DEFAULT_VISUALIZATIONUTILITIES_HPP