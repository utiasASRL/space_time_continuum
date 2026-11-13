#ifndef DEFAULT_SENSORVIEWS_HPP
#define DEFAULT_SENSORVIEWS_HPP

#include <memory>

#include <spacetime/estimator/types/SensorBase.hpp>
#include <spacetime/visualizer/default/Visualizable.hpp>
#include <spacetime/visualizer/default/objects/PoseSensor.hpp>

namespace Spacetime::Visualization::Default
{
    /**
     * @brief The default-backend view of `sensor`, chosen from its dynamic type, for
     *        a robot whose node values are ValuesT. Sensors without a view of their
     *        own (e.g. Sensors::GyroSensor) get a Visualizable that draws nothing.
     *        Views subscribe to the sensor's measurements (see MeasurementHistory),
     *        so call this before they are consumed.
     */
    template <typename ValuesT>
    std::shared_ptr<Visualizable> makeSensorView(const std::shared_ptr<Types::SensorBase> &sensor)
    {
        if constexpr (requires { typename Sensors::PoseSensor<ValuesT>; })
            if (auto pose_sensor = std::dynamic_pointer_cast<Sensors::PoseSensor<ValuesT>>(sensor))
                return std::make_shared<PoseSensor<ValuesT>>(pose_sensor);
        return std::make_shared<Visualizable>();
    }
} // namespace Spacetime::Visualization::Default

#endif // DEFAULT_SENSORVIEWS_HPP
