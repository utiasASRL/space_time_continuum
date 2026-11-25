#ifndef GROUNDPLANE_HPP
#define GROUNDPLANE_HPP

#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <vtkActor.h>
#include <vtkPlaneSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkSmartPointer.h>

#include <spacetime/visualizer/VisualizerOptions.hpp>
#include <spacetime/visualizer/default/Visualizable.hpp>
#include <spacetime/visualizer/default/Visualizer.hpp>

namespace Spacetime::Visualization::Default
{
    /**
     * @brief A flat light-grey square: the scene's floor, drawn only when a program adds it
     *        (Visualizer::addVisualizable). Its centre, normal and side length come from
     *        GroundPlaneOptions. Its in-plane edges run along world x projected onto the plane
     *        (world y if x is along the normal), so a +z normal gives an axis-aligned square.
     *        Static (never stale); P toggles it unless setToggleKey picks another key.
     */
    class GroundPlane : public Visualizable
    {
    public:
        /// The four corners of the square: origin, point 1 and point 2 of its vtkPlaneSource, and the opposite corner.
        struct Corners
        {
            Eigen::Vector3d origin, point1, point2, opposite;
        };

        /// @throws std::invalid_argument if the normal is zero or not finite, the position is not finite, or the size is not positive and finite.
        explicit GroundPlane(const GroundPlaneOptions &options = GroundPlaneOptions()) : Visualizable(/*is_static=*/true), m_options(validated(options))
        {
            toggle_key_ = Qt::Key_P;
        }

        /// @see GroundPlane(const GroundPlaneOptions &)
        GroundPlane(const Eigen::Vector3d &normal, const Eigen::Vector3d &position, double size)
            : GroundPlane(GroundPlaneOptions{normal, position, size}) {}

        /// "GroundPlane", whether or not `brief`.
        std::string info(bool brief = true) const override { return "GroundPlane"; }

        /// The options it was built with, the normal normalized.
        const GroundPlaneOptions &options() const { return m_options; }

        /// The square's corners in the world frame (see the class description).
        Corners corners() const
        {
            const Eigen::Vector3d &n = m_options.normal;
            Eigen::Vector3d u = Eigen::Vector3d::UnitX() - n.x() * n;
            if (u.norm() < 1e-6)
                u = Eigen::Vector3d::UnitY() - n.y() * n;
            u.normalize();
            const Eigen::Vector3d v = n.cross(u);
            const Eigen::Vector3d &c = m_options.position;
            const double h = 0.5 * m_options.size;
            return {c - h * (u + v), c + h * (u - v), c + h * (v - u), c + h * (u + v)};
        }

        /**
         * @brief Creates the plane's actor (once; later calls return the same one), shown
         *        or hidden as set by setVisible. Needs no Visualizer, so it can be tested
         *        without a window.
         */
        vtkActor *createActor() const
        {
            if (mp_actor)
                return mp_actor;
            Corners corners = this->corners(); // non-const: vtkPlaneSource::SetPoint1/2 take double *
            vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();
            plane->SetOrigin(corners.origin.data());
            plane->SetPoint1(corners.point1.data());
            plane->SetPoint2(corners.point2.data());

            vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            mapper->SetInputConnection(plane->GetOutputPort());

            mp_actor = vtkSmartPointer<vtkActor>::New();
            mp_actor->SetMapper(mapper);
            mp_actor->GetProperty()->SetColor(0.9, 0.9, 0.9);
            mp_actor->GetProperty()->SetAmbient(0.3);
            mp_actor->GetProperty()->SetDiffuse(0.5);
            mp_actor->GetProperty()->SetSpecular(0.1);
            mp_actor->SetVisibility(visible_);
            return mp_actor;
        }

        /// The actor, or null before createActor / visualize.
        vtkActor *actor() const { return mp_actor; }

        /// Adds the plane's actor to `visualizer` and binds its visibility toggle (P by default). Nothing is updated per frame.
        void visualize(Visualizer &visualizer) const override
        {
            visualizer.addActor(createActor());
            bindVisibilityToggle(visualizer, "ground plane");
        }

        /// Flips the visibility, and the actor's with it.
        void toggleVisibility() const override
        {
            visible_ = !visible_;
            if (mp_actor)
                mp_actor->SetVisibility(visible_);
        }

    private:
        GroundPlaneOptions m_options;
        mutable vtkSmartPointer<vtkActor> mp_actor;

        static GroundPlaneOptions validated(GroundPlaneOptions options)
        {
            if (!options.normal.allFinite() || options.normal.norm() < 1e-12)
                throw std::invalid_argument("GroundPlane: the normal must be nonzero and finite.");
            if (!options.position.allFinite())
                throw std::invalid_argument("GroundPlane: the position must be finite.");
            if (!(options.size > 0.0 && std::isfinite(options.size)))
                throw std::invalid_argument("GroundPlane: the size must be positive and finite.");
            options.normal.normalize();
            return options;
        }
    };
} // namespace Spacetime::Visualization::Default
#endif // GROUNDPLANE_HPP
