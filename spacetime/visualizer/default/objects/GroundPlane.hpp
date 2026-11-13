#ifndef GROUNDPLANE_HPP
#define GROUNDPLANE_HPP

#include <vtkSmartPointer.h>
#include <vtkRegularPolygonSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>

#include <spacetime/visualizer/default/Visualizable.hpp>
#include <spacetime/visualizer/default/Visualizer.hpp>

namespace Spacetime::Visualization::Default
{
    /// The scene's flat decorative floor: a light-grey square (circumradius 0.5, rotated 45 deg about z) in the z = 0 plane.
    /// Static, so it is never hidden as stale.
    class GroundPlane : public Visualizable
    {
    public:
        /// Constructs a static (never stale) view.
        GroundPlane() : Visualizable(/*is_static=*/true) {}

        /// "GroundPlane", whether or not `brief`.
        std::string info(bool brief = true) const override { return "GroundPlane"; }

        /// Adds the floor actor to `visualizer`. Nothing is updated per frame.
        void visualize(Visualizer &visualizer) const override
        {
            vtkSmartPointer<vtkRegularPolygonSource> floorPoly = vtkSmartPointer<vtkRegularPolygonSource>::New();
            floorPoly->SetNumberOfSides(4);
            floorPoly->SetNormal(0, 0, 1);
            floorPoly->SetCenter(0, 0, 0.0);
            floorPoly->SetRadius(0.5);

            vtkSmartPointer<vtkPolyDataMapper> floorMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            floorMapper->SetInputConnection(floorPoly->GetOutputPort());

            vtkSmartPointer<vtkActor> floor = vtkSmartPointer<vtkActor>::New();
            floor->SetMapper(floorMapper);
            floor->GetProperty()->SetColor(0.9, 0.9, 0.9);
            floor->GetProperty()->SetAmbient(0.3);
            floor->GetProperty()->SetDiffuse(0.5);
            floor->GetProperty()->SetSpecular(0.1);
            floor->RotateZ(45.0);

            visualizer.addActor(floor);
        }
    };
} // namespace Spacetime::Visualization::Default
#endif // GROUNDPLANE_HPP
