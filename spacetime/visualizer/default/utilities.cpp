#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkCylinderSource.h>
#include <vtkActor.h>
#include <vtkAppendPolyData.h>
#include <vtkConeSource.h>
#include <vtkPointData.h>
#include <vtkPolyDataAlgorithm.h>
#include <vtkPolyDataNormals.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkSTLReader.h>

#include "utilities.hpp"

namespace Spacetime::Visualization::Default::Utilities
{
    vtkSmartPointer<vtkPolyDataMapper> createFrameMapper(double cylinderRadius, double totalLength)
    {
        const double shaft_length = 0.8 * totalLength, tip_length = totalLength - shaft_length;

        // One axis along +y, as vtkAxesActor builds it before turning it onto x and z.
        vtkSmartPointer<vtkCylinderSource> shaft = vtkSmartPointer<vtkCylinderSource>::New();
        shaft->SetRadius(cylinderRadius * shaft_length);
        shaft->SetHeight(shaft_length);
        shaft->SetCenter(0.0, 0.5 * shaft_length, 0.0);
        shaft->SetResolution(kFrameResolution);
        vtkSmartPointer<vtkConeSource> tip = vtkSmartPointer<vtkConeSource>::New();
        tip->SetDirection(0.0, 1.0, 0.0);
        tip->SetRadius(0.4 * tip_length);
        tip->SetHeight(tip_length);
        tip->SetCenter(0.0, shaft_length + 0.5 * tip_length, 0.0);
        tip->SetResolution(kFrameResolution);

        vtkSmartPointer<vtkAppendPolyData> frame = vtkSmartPointer<vtkAppendPolyData>::New();
        for (int axis = 0; axis < 3; axis++)
        {
            vtkSmartPointer<vtkTransform> onto_axis = vtkSmartPointer<vtkTransform>::New();
            if (axis == 0)
                onto_axis->RotateZ(-90.0); // +y onto +x
            else if (axis == 2)
                onto_axis->RotateX(90.0); // +y onto +z
            for (vtkPolyDataAlgorithm *part : {static_cast<vtkPolyDataAlgorithm *>(shaft), static_cast<vtkPolyDataAlgorithm *>(tip)})
            {
                vtkSmartPointer<vtkTransformPolyDataFilter> placed = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
                placed->SetInputConnection(part->GetOutputPort());
                placed->SetTransform(onto_axis);
                placed->Update();

                vtkSmartPointer<vtkPolyData> piece = vtkSmartPointer<vtkPolyData>::New();
                piece->ShallowCopy(placed->GetOutput());
                piece->GetPointData()->Initialize(); // the shaft's normals and texture coordinates: the normals are recomputed below for the whole mesh
                vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
                colors->SetName("Colors");
                colors->SetNumberOfComponents(3);
                colors->SetNumberOfTuples(piece->GetNumberOfPoints());
                for (vtkIdType i = 0; i < piece->GetNumberOfPoints(); i++)
                    colors->SetTuple3(i, axis == 0 ? 255 : 0, axis == 1 ? 255 : 0, axis == 2 ? 255 : 0);
                piece->GetPointData()->SetScalars(colors);
                frame->AddInputData(piece);
            }
        }

        // Smooth along the shafts and cones, split at their caps and bases (as the sources shade them).
        vtkSmartPointer<vtkPolyDataNormals> normals = vtkSmartPointer<vtkPolyDataNormals>::New();
        normals->SetInputConnection(frame->GetOutputPort());
        normals->Update();
        vtkSmartPointer<vtkPolyData> mesh = vtkSmartPointer<vtkPolyData>::New();
        mesh->ShallowCopy(normals->GetOutput());

        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputData(mesh);
        mapper->ScalarVisibilityOn();
        mapper->SetScalarModeToUsePointData();
        mapper->SetColorModeToDirectScalars();
        return mapper;
    }

    void applySurfaceMaterial(vtkProperty *property)
    {
        property->SetAmbient(kSurfaceAmbient);
        property->SetDiffuse(kSurfaceDiffuse);
        property->SetSpecular(kSurfaceSpecular);
        property->SetSpecularPower(kSurfaceSpecularPower);
    }

    vtkSmartPointer<vtkActor> createFrameActor(vtkSmartPointer<vtkPolyDataMapper> mapper)
    {
        vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->SetVisibility(true);
        return actor;
    }

    vtkSmartPointer<vtkActor> createFrameActor(double cylinderRadius, double totalLength)
    {
        return createFrameActor(createFrameMapper(cylinderRadius, totalLength));
    }

    vtkSmartPointer<vtkActor> createEllipsoidActor(vtkSmartPointer<vtkPolyDataMapper> mapper, bool red)
    {
        vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        if (red)
            actor->GetProperty()->SetColor(1, 0, 0);
        else
            actor->GetProperty()->SetColor(0, 0, 1);
        actor->GetProperty()->SetOpacity(0.15);
        actor->GetProperty()->SetAmbient(0.3);
        actor->GetProperty()->SetDiffuse(0.5);
        actor->GetProperty()->SetSpecular(0.1);
        actor->SetVisibility(false);
        return actor;
    }

    vtkSmartPointer<vtkPolyDataMapper> createUnitSphereMapper()
    {
        vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();
        sphere->SetRadius(1.0);
        sphere->SetThetaResolution(kSphereResolution);
        sphere->SetPhiResolution(kSphereResolution);
        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputConnection(sphere->GetOutputPort());
        return mapper;
    }

    vtkSmartPointer<vtkPolyDataMapper> createUnitCylinderMapper()
    {
        vtkSmartPointer<vtkCylinderSource> cylinder = vtkSmartPointer<vtkCylinderSource>::New();
        cylinder->SetRadius(1.0);
        cylinder->SetHeight(1.0);
        cylinder->SetResolution(kCylinderResolution);
        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputConnection(cylinder->GetOutputPort());
        return mapper;
    }

    vtkSmartPointer<vtkMatrix4x4> getEllipsoidMatrix(const Eigen::Vector3d &position, const Eigen::Matrix3d &covariance, double scale)
    {
        return toVtkMatrix(ellipsoidTransform(position, covariance, scale));
    }

    Eigen::Matrix4d cylinderMatrixBetween(const Eigen::Vector3d &p0, const Eigen::Vector3d &p1, double radius)
    {
        const Eigen::Vector3d segment = p1 - p0;
        const double length = segment.norm();
        // Rotation taking the local y axis onto the segment (any rotation for a degenerate one).
        const Eigen::Matrix3d rotation = length > 0.0
                                             ? Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitY(), segment).toRotationMatrix()
                                             : Eigen::Matrix3d::Identity();
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = rotation * Eigen::Vector3d(radius, length, radius).asDiagonal();
        T.block<3, 1>(0, 3) = 0.5 * (p0 + p1);
        return T;
    }

    vtkSmartPointer<vtkMatrix4x4> toVtkMatrix(const Eigen::Matrix4d &matrix)
    {
        vtkSmartPointer<vtkMatrix4x4> out = vtkSmartPointer<vtkMatrix4x4>::New();
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                out->SetElement(r, c, matrix(r, c));
        return out;
    }

    void orbitAboutVertical(vtkCamera *camera, double angle)
    {
        if (angle == 0.0)
            return;
        const Eigen::Matrix3d R = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        const Eigen::Vector3d focal_point(camera->GetFocalPoint());
        const Eigen::Vector3d position = focal_point + R * (Eigen::Vector3d(camera->GetPosition()) - focal_point);
        const Eigen::Vector3d view_up = R * Eigen::Vector3d(camera->GetViewUp());
        camera->SetPosition(position.data());
        camera->SetViewUp(view_up.data());
    }

    vtkSmartPointer<vtkPolyData> loadSTLMesh(const std::string &path, double scale)
    {
        if (!std::filesystem::is_regular_file(path))
            throw std::runtime_error("loadSTLMesh: mesh file not found: " + path);

        vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
        reader->SetFileName(path.c_str());
        vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
        transform->Scale(scale, scale, scale);
        vtkSmartPointer<vtkTransformPolyDataFilter> filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
        filter->SetInputConnection(reader->GetOutputPort());
        filter->SetTransform(transform);
        filter->Update();

        vtkSmartPointer<vtkPolyData> mesh = vtkSmartPointer<vtkPolyData>::New();
        mesh->ShallowCopy(filter->GetOutput());
        if (mesh->GetNumberOfPoints() == 0)
            throw std::runtime_error("loadSTLMesh: no geometry read from " + path);
        return mesh;
    }

    void setUserMatrix(vtkProp3D *prop, const Eigen::Matrix4d &matrix)
    {
        vtkMatrix4x4 *user = prop->GetUserMatrix();
        if (!user)
        {
            prop->SetUserMatrix(toVtkMatrix(matrix));
            return;
        }
        // SetElement only marks the matrix modified when a value changes, and the prop's
        // MTime includes its user matrix's, so the prop picks the new pose up by itself.
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                user->SetElement(r, c, matrix(r, c));
    }
} // namespace Spacetime::Visualization::Default::Utilities
