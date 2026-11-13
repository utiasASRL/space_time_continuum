#ifndef ROBOT_HPP
#define ROBOT_HPP

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <spacetime/estimator/types/Model.hpp>

namespace Spacetime::Types
{
    /**
     * @brief A convenience list of the models making up one robot (e.g. its prior and
     *        sensors), added to an estimator together with Estimator::addRobot.
     */
    struct Robot
    {
        /// Appends `model`. @throws std::invalid_argument for a null model.
        void addModel(pModelType model)
        {
            if (!model)
                throw std::invalid_argument("Robot::addModel: the model is null.");
            m_models.push_back(std::move(model));
        }

        /// The models added so far, in the order added.
        const std::vector<pModelType> &models() const { return m_models; }

    private:
        std::vector<pModelType> m_models;
    };
} // namespace Spacetime::Types
#endif // ROBOT_HPP
