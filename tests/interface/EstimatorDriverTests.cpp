#include <memory>
#include <stdexcept>
#include <vector>

#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/interface/EstimatorDriver.hpp>
#include "../TestEstimatorOptions.hpp"

namespace Spacetime::Interface
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::NodeValues<Pose>;

        std::shared_ptr<Priors::WNOA::Prior<Pose>> makePrior()
        {
            auto state = std::make_shared<StateType>(1.0, 2);
            return std::make_shared<Priors::WNOA::Prior<Pose>>(state, ValueType());
        }
    } // namespace

    TEST(EstimatorDriver, StepEstimatesAtTheTimeThenCallsTheCallback)
    {
        Estimator estimator(Testing::estimatorOptions(0.0));
        estimator.addPrior(makePrior());
        EstimatorDriver driver(estimator, /*real_time=*/false);

        std::vector<double> ticks;
        std::vector<double> snapshot_times; // the snapshot the callback sees is already this step's
        driver.setTickCallback([&](double t)
                               {
            ticks.push_back(t);
            snapshot_times.push_back(estimator.getSnapshot()->time); });

        driver.step(0.0);
        driver.step(0.1);
        EXPECT_EQ(ticks, (std::vector<double>{0.0, 0.1}));
        EXPECT_EQ(snapshot_times, (std::vector<double>{0.0, 0.1}));
    }

    TEST(EstimatorDriver, FinishSolvesABatchEstimator)
    {
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG));
        estimator.addPrior(makePrior());
        EstimatorDriver driver(estimator, false);
        driver.step(0.0);
        driver.step(0.1);
        EXPECT_EQ(estimator.getSnapshot(), nullptr); // only accumulated so far
        driver.finish();
        EXPECT_NE(estimator.getSnapshot(), nullptr);
    }

    TEST(EstimatorDriver, RejectsARealTimeBatchEstimator)
    {
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG));
        EXPECT_THROW(EstimatorDriver(estimator, /*real_time=*/true), std::invalid_argument);
        EXPECT_NO_THROW(EstimatorDriver(estimator, /*real_time=*/false));
    }
} // namespace Spacetime::Interface
