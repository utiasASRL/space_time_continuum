#ifndef DEFAULT_POSESENSOR_HPP
#define DEFAULT_POSESENSOR_HPP

#include <memory>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkMatrix4x4.h>

#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/visualizer/MeasurementHistory.hpp>
#include <spacetime/visualizer/default/Visualizable.hpp>
#include <spacetime/visualizer/default/Visualizer.hpp>
#include <spacetime/visualizer/default/utilities.hpp>

namespace Spacetime::Visualization::Default
{
    /// Default-backend view of a Sensors::PoseSensor<ValuesT>: a frame at its last received measurement's pose.
    template <typename ValuesT>
    class PoseSensor : public Visualizable
    {
        using Pose = typename ValuesT::PoseType;

    public:
        /// Subscribes to `sensor`'s measurements (see MeasurementHistory), so construct this before they are consumed.
        explicit PoseSensor(const std::shared_ptr<Sensors::PoseSensor<ValuesT>> &sensor)
            : m_history(requireSensor(sensor)) {}

        /// Short name if `brief`, otherwise a one-line description.
        std::string info(bool brief = true) const override { return brief ? "PoseSensor" : "PoseSensor: last received measurement's pose"; }

        /// Adds the (initially hidden) measurement frame and binds its visibility toggle, if any.
        void visualize(Visualizer &visualizer) const override
        {
            mp_axes = Utilities::createFrameActor(visualizer.frameMapper(0.01, 0.05));
            mp_axes->SetVisibility(false);
            visualizer.addActor(mp_axes);
            bindVisibilityToggle(visualizer, info(true));
        }

        /// Places the frame at the latest measurement at or before `time` (its value is used directly as the drawn T_ib pose); hidden if there is none, it is stale, or toggled off.
        void updateVisualization(double time, double /*estimate_time*/) const override
        {
            auto measurement = m_history.latestMeasurementAt(time);
            if (!measurement)
            {
                mp_axes->SetVisibility(false);
                return;
            }
            Utilities::setUserMatrix(mp_axes, Pose(measurement->value).matrix());
            mp_axes->SetVisibility(visible_ && isFresh(time, measurement->time));
        }

        /// Frame drawn at the measured pose; created by visualize().
        mutable vtkSmartPointer<vtkActor> mp_axes;

    private:
        /// `*sensor`, or throws std::invalid_argument if it is null (so the check runs before m_history subscribes).
        static Types::SensorBase &requireSensor(const std::shared_ptr<Sensors::PoseSensor<ValuesT>> &sensor)
        {
            if (!sensor)
                throw std::invalid_argument("Default::PoseSensor requires a non-null sensor.");
            return *sensor;
        }

        /// Every measurement the sensor received, for lookup at any playback time.
        MeasurementHistory m_history;
    };
} // namespace Spacetime::Visualization::Default

#endif // DEFAULT_POSESENSOR_HPP
