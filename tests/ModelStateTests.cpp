#include <concepts>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/joint_models/EndEffector.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/estimator/types/Robot.hpp>

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::Values<Pose>;
        using PriorType = Priors::WNOA::Prior<Pose>;
        using Factors = Types::Model::Factors;

        std::shared_ptr<StateType> makeState() { return std::make_shared<StateType>(1.0, 3); }

        std::shared_ptr<PriorType> makePrior(std::shared_ptr<StateType> state) { return std::make_shared<PriorType>(std::move(state), ValueType()); }

        std::shared_ptr<Sensors::PoseSensor<ValueType>> makeSensor(const std::shared_ptr<PriorType> &prior)
        {
            return std::make_shared<Sensors::PoseSensor<ValueType>>(Eigen::Matrix<double, 6, 6>::Identity(), prior);
        }

        /// A stateful model extending its state by one row per call, relying on StatefulModel's default outer factors and queries.
        class RowModel : public Types::StatefulModel<StateType>
        {
        public:
            using Types::StatefulModel<StateType>::StatefulModel;

            Factors extendState(KeyManager &km, double time, const gtsam::Values &, gtsam::Values &) override
            {
                state()->addStateAtTime(km, time);
                return {};
            }
        };

        template <typename T>
        concept HasState = requires(const T &model) { model.state(); };

        /// A stateless model logging its extendState calls into `log`.
        class RecordingModel : public Types::Model
        {
        public:
            RecordingModel(std::string name, std::vector<std::string> &log) : m_name(std::move(name)), m_log(log) {}

            Factors extendState(KeyManager &, double, const gtsam::Values &, gtsam::Values &) override
            {
                m_log.push_back(m_name);
                return {};
            }

        private:
            std::string m_name;
            std::vector<std::string> &m_log;
        };
    } // namespace

    TEST(Model, IsStateless)
    {
        EXPECT_FALSE(HasState<Types::Model>);
        EXPECT_TRUE(HasState<Types::StatefulModel<>>);
    }

    TEST(StatefulModel, StoresItsTypedState)
    {
        auto state = makeState();
        RowModel model(state);
        static_assert(std::same_as<decltype(model.state()), std::shared_ptr<StateType>>);
        EXPECT_EQ(model.state(), state);

        // The same state, untyped, through the common base sensors hold.
        const Types::StatefulModel<> &base = model;
        EXPECT_EQ(base.state(), state);
    }

    TEST(StatefulModel, NullStateThrows)
    {
        EXPECT_THROW(RowModel(nullptr), std::invalid_argument);
        EXPECT_THROW(makePrior(nullptr), std::invalid_argument);
    }

    TEST(StatefulModel, Hierarchy)
    {
        EXPECT_TRUE((std::derived_from<Types::StatefulModel<>, Types::Model>));
        EXPECT_TRUE((std::derived_from<Types::StatefulModel<StateType>, Types::StatefulModel<>>));
        EXPECT_TRUE((std::derived_from<PriorType, Types::StatefulModel<StateType>>));
        EXPECT_TRUE((std::derived_from<Joints::EndEffector<Pose>, Types::StatefulModel<>>));
        EXPECT_TRUE((std::derived_from<Types::SensorBase, Types::Model>));
        EXPECT_FALSE((std::derived_from<Types::SensorBase, Types::StatefulModel<>>));
        EXPECT_TRUE((std::derived_from<Types::ActuatorBase, Types::Model>));

        EXPECT_TRUE(Types::StatefulType<PriorType>);
        EXPECT_TRUE(Types::StatefulType<Joints::EndEffector<Pose>>);
        EXPECT_FALSE(Types::StatefulType<Types::SensorBase>);
    }

    TEST(StatefulModel, PriorStoresTheStateItExtends)
    {
        auto state = makeState();
        auto prior = makePrior(state);
        EXPECT_EQ(prior->state(), state);

        KeyManager km;
        gtsam::Values new_values;
        prior->extendState(km, 0.0, gtsam::Values(), new_values);
        EXPECT_EQ(state->estimation_nodes.size(), 3u);

        using WNOVNode = Priors::WNOV::Node<Pose>;
        auto wnov_state = std::make_shared<Types::ContinuumRobotState<WNOVNode>>(1.0, 3);
        EXPECT_EQ(std::make_shared<Priors::WNOV::Prior<Pose>>(wnov_state, Pose(), gtsam::Vector6::Zero())->state(), wnov_state);
    }

    TEST(StatefulModel, ItsLockedValuesAreItsStates)
    {
        auto state = makeState();
        state->lock("pose", 0.0, Pose());
        RowModel model(state);

        KeyManager km;
        gtsam::Values new_values;
        model.extendState(km, 0.0, gtsam::Values(), new_values);
        EXPECT_EQ(model.lockedValues().size(), 1u);
        EXPECT_EQ(&model.lockedValues(), &state->lockedValues());
    }

    TEST(StatefulModel, SensorsGenerateTheirOuterFactorsWithIt)
    {
        auto prior = makePrior(makeState());
        auto sensor = makeSensor(prior);
        EXPECT_EQ(sensor->stateful(), prior);
        EXPECT_THROW(Sensors::PoseSensor<ValueType>(Eigen::Matrix<double, 6, 6>::Identity(), nullptr), std::runtime_error);
    }

    TEST(Robot, KeepsItsModelsInOrder)
    {
        auto prior = makePrior(makeState());
        auto sensor = makeSensor(prior);
        Types::Robot robot;
        robot.addModel(prior);
        robot.addModel(sensor);
        ASSERT_EQ(robot.models().size(), 2u);
        EXPECT_EQ(robot.models()[0], prior);
        EXPECT_EQ(robot.models()[1], sensor);
    }

    TEST(Robot, NullModelThrows)
    {
        Types::Robot robot;
        EXPECT_THROW(robot.addModel(nullptr), std::invalid_argument);
        EXPECT_TRUE(robot.models().empty());
    }

    TEST(EstimatorAddRobot, AddsTheRobotsModelsInOrder)
    {
        std::vector<std::string> log;
        auto prior = makePrior(makeState());
        auto sensor = makeSensor(prior);
        Types::Robot robot;
        robot.addModel(std::make_shared<RecordingModel>("a", log));
        robot.addModel(prior);
        robot.addModel(sensor);
        robot.addModel(std::make_shared<RecordingModel>("b", log));

        Estimator estimator;
        estimator.addModel(std::make_shared<RecordingModel>("first", log));
        estimator.addRobot(robot);
        ASSERT_EQ(estimator.getMeasurementQueues().size(), 1u);
        EXPECT_EQ(estimator.getMeasurementQueues()[0], sensor);

        estimator.estimate(0.0);
        EXPECT_EQ(log, (std::vector<std::string>{"first", "a", "b"}));
        EXPECT_FALSE(estimator.getCurrentValues().empty()); // the robot's prior was extended
    }

    TEST(EstimatorAddModel, SecondStatefulModelOnTheSameStateThrows)
    {
        auto state = makeState();
        auto prior = makePrior(state);
        Estimator estimator;
        estimator.addPrior(prior);
        EXPECT_THROW(estimator.addPrior(makePrior(state)), std::invalid_argument);
        EXPECT_THROW(estimator.addModel(std::make_shared<RowModel>(state)), std::invalid_argument);
        EXPECT_THROW(estimator.addPrior(prior), std::invalid_argument); // the same prior twice

        std::vector<std::string> log;
        Types::Robot robot;
        robot.addModel(std::make_shared<RecordingModel>("valid", log));
        robot.addModel(makePrior(state));
        EXPECT_THROW(estimator.addRobot(robot), std::invalid_argument);

        // Nothing rejected was added, not even the robot's valid model: the state is extended once per step.
        estimator.estimate(0.0);
        EXPECT_EQ(state->estimation_nodes.size(), 3u);
        EXPECT_TRUE(log.empty());
    }

    TEST(EstimatorAddModel, StatefulModelsOnDifferentStatesAreAccepted)
    {
        Estimator estimator;
        estimator.addPrior(makePrior(makeState()));
        EXPECT_NO_THROW(estimator.addPrior(makePrior(makeState())));
    }

    // An end effector extends its state through its own prior, so the prior is not added separately.
    TEST(EstimatorAddModel, AnEndEffectorAndItsOwnPriorCannotBothBeAdded)
    {
        auto ee = std::make_shared<Joints::EndEffector<Pose>>(Pose());
        Estimator estimator;
        estimator.addModel(ee);
        EXPECT_THROW(estimator.addModel(ee->endEffectorPrior()), std::invalid_argument);
    }
} // namespace Spacetime
