#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtest/gtest.h>

#include <spacetime/estimator/types/ActuatorState.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

namespace Spacetime
{
    TEST(ActuatorState, RejectsAZeroInputDim)
    {
        EXPECT_THROW(Types::ActuatorState(0), std::invalid_argument);
        const Types::ActuatorState state(3, 'w');
        EXPECT_EQ(state.input_dim, 3u);
        EXPECT_EQ(state.input_symbol, 'w');
        EXPECT_TRUE(state.inputs().empty());
        EXPECT_TRUE(state.times().empty());
    }

    // Each input is a node at its measurement's time, holding one variable keyed by the state's symbol.
    TEST(ActuatorState, AddInputAddsOneNodePerInputInTimeOrder)
    {
        Types::ActuatorState state(2, 'u');
        KeyManager km;
        const auto first = state.addInput(km, 0.2);
        const auto earlier = state.addInput(km, 0.1); // a late measurement still lands in time order
        const auto last = state.addInput(km, 0.3);

        ASSERT_EQ(state.inputs().size(), 3u);
        EXPECT_EQ(state.inputs()[0], earlier);
        EXPECT_EQ(state.inputs()[1], first);
        EXPECT_EQ(state.inputs()[2], last);
        EXPECT_EQ(state.times(), (std::vector<double>{0.1, 0.2, 0.3}));

        EXPECT_EQ(first->inputKey(), gtsam::Symbol('u', 0));
        EXPECT_EQ(earlier->inputKey(), gtsam::Symbol('u', 1));
        EXPECT_DOUBLE_EQ(first->time, 0.2);
        EXPECT_DOUBLE_EQ(first->arclength, 0.0);
        EXPECT_EQ(first->keys(), gtsam::KeyVector{first->inputKey()});
        EXPECT_DOUBLE_EQ(km.drainPending().at(first->inputKey()), 0.2); // timestamped for the smoother
    }

    TEST(ActuatorState, InputNodeMapsTheInterpolatedInputKeyAndReadsItsValue)
    {
        Types::ActuatorState state(2);
        KeyManager km;
        const auto node = state.addInput(km, 0.0);
        const auto keys = node->interpolatedKeys();
        ASSERT_EQ(keys.size(), 1u);
        EXPECT_EQ(keys[0], std::make_pair(Types::InputKeys::Input(), node->inputKey()));

        gtsam::Values values;
        values.insert(node->inputKey(), gtsam::Vector(Eigen::Vector2d(1.0, -2.0)));
        const auto read = std::dynamic_pointer_cast<Types::InputValues>(node->getValuesBase(values));
        ASSERT_NE(read, nullptr);
        EXPECT_TRUE(read->input().isApprox(Eigen::Vector2d(1.0, -2.0)));
    }

    // Every input node has its state's input_dim; its values start at zero, and a value of
    // another size cannot be read as it.
    TEST(ActuatorState, InputNodesHaveTheStatesInputDim)
    {
        Types::ActuatorState state(3);
        KeyManager km;
        const auto node = state.addInput(km, 0.0);
        EXPECT_EQ(node->dim(), 3u);

        const Types::InputValues initial(3);
        ASSERT_EQ(initial.values.size(), 1u); // one component, the input
        EXPECT_EQ(initial.input().rows(), 3);
        EXPECT_EQ(initial.input().cols(), 1);
        EXPECT_TRUE(initial.input().isZero());

        gtsam::Values wrong_size;
        wrong_size.insert(node->inputKey(), gtsam::Vector(Eigen::Vector2d(1.0, 2.0)));
        EXPECT_THROW(node->getValuesBase(wrong_size), std::runtime_error);
    }

    // The nearest input in time, whatever the arclength; ties go to the earlier input, and
    // times outside the inputs' extent have none.
    TEST(ActuatorState, NearestNodeIsTheNearestInputInTime)
    {
        Types::ActuatorState state(1);
        EXPECT_EQ(state.getNearestNode(0.0, 0.0), nullptr);

        KeyManager km;
        const auto a = state.addInput(km, 0.0), b = state.addInput(km, 0.1), c = state.addInput(km, 0.3);
        EXPECT_EQ(state.getNearestNode(0.5, 0.0), a);
        EXPECT_EQ(state.getNearestNode(0.0, 0.04), a);
        EXPECT_EQ(state.getNearestNode(0.0, 0.05), a); // tie: the earlier one
        EXPECT_EQ(state.getNearestNode(0.0, 0.06), b);
        EXPECT_EQ(state.getNearestNode(0.0, 0.25), c);
        EXPECT_EQ(state.getNearestNode(0.0, 0.3), c);
        EXPECT_EQ(state.getNearestNode(0.0, -0.01), nullptr);
        EXPECT_EQ(state.getNearestNode(0.0, 0.31), nullptr);
    }
} // namespace Spacetime
