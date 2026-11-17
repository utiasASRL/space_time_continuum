#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <spacetime/visualizer/default/Visualizable.hpp>
#include <spacetime/visualizer/default/KeyBindingRegistry.hpp>

namespace Spacetime
{
    namespace
    {
        struct TestVisualizable : public Visualization::Default::Visualizable
        {
            bool isVisible() const { return visible_; }

            template <typename VisualizerT>
            void bindToggle(VisualizerT &visualizer) const { bindVisibilityToggle(visualizer, "Test"); }
        };

        // Mirrors how Visualization::Default::Visualizer::bindToggleShortcut drives its
        // KeyBindingRegistry (counting the QShortcuts / panel rows it would
        // create), so several Visualizables sharing a key can be checked
        // end-to-end without a real Qt/VTK Visualizer.
        struct RegistryBackedVisualizer
        {
            Visualization::Default::KeyBindingRegistry registry;
            int shortcuts_created = 0;
            std::vector<std::string> panel_rows;

            void bindToggleShortcut(Qt::Key key, std::function<void()> toggleFn, std::function<bool()> stateFn, const std::string &label)
            {
                if (registry.addHandler(key, std::move(toggleFn)))
                    ++shortcuts_created;
                if (registry.addToggleState(key, std::move(stateFn)))
                    panel_rows.push_back(label);
            }
        };
    }

    TEST(KeyBindingRegistry, HandlersAndStatesAreTrackedIndependently)
    {
        // A plain bindShortcut on a key shouldn't stop a later toggle on that key from getting its panel row.
        Visualization::Default::KeyBindingRegistry registry;
        EXPECT_TRUE(registry.addHandler(Qt::Key_S, []() {}));
        EXPECT_TRUE(registry.addToggleState(Qt::Key_S, []() { return true; }));
    }

    TEST(KeyBindingRegistry, DispatchRunsEveryHandlerForTheKeyInOrder)
    {
        Visualization::Default::KeyBindingRegistry registry;
        std::vector<int> calls;
        registry.addHandler(Qt::Key_S, [&]() { calls.push_back(1); });
        registry.addHandler(Qt::Key_G, [&]() { calls.push_back(99); });
        registry.addHandler(Qt::Key_S, [&]() { calls.push_back(2); });
        registry.addHandler(Qt::Key_S, [&]() { calls.push_back(3); });

        registry.dispatch(Qt::Key_S);
        EXPECT_EQ(calls, (std::vector<int>{1, 2, 3}));

        registry.dispatch(Qt::Key_G);
        EXPECT_EQ(calls, (std::vector<int>{1, 2, 3, 99}));
    }

    TEST(KeyBindingRegistry, DispatchOfAnUnregisteredKeyIsANoOp)
    {
        Visualization::Default::KeyBindingRegistry registry;
        bool called = false;
        registry.addHandler(Qt::Key_S, [&]() { called = true; });
        EXPECT_NO_THROW(registry.dispatch(Qt::Key_G));
        EXPECT_FALSE(called);

        Visualization::Default::KeyBindingRegistry empty;
        EXPECT_NO_THROW(empty.dispatch(Qt::Key_S));
    }

    TEST(KeyBindingRegistry, AllOnIsVacuouslyTrueForAKeyWithNoStates)
    {
        Visualization::Default::KeyBindingRegistry registry;
        EXPECT_TRUE(registry.allOn(Qt::Key_S));
        registry.addHandler(Qt::Key_S, []() {}); // a handler alone adds no state
        EXPECT_TRUE(registry.allOn(Qt::Key_S));
    }

    // The regression this registry fixes: several Visualizables bound to one key
    // (e.g. every PoseSensor on Qt::Key_S) used to each get their own QShortcut,
    // which Qt treats as ambiguous and never activates.
    TEST(KeyBindingRegistry, SharedKeyTogglesEveryVisualizableTogetherWithOneShortcutAndRow)
    {
        RegistryBackedVisualizer visualizer;
        std::vector<std::unique_ptr<TestVisualizable>> sensors;
        for (int i = 0; i < 3; ++i)
        {
            sensors.push_back(std::make_unique<TestVisualizable>());
            sensors.back()->setToggleKey(Qt::Key_S);
            sensors.back()->bindToggle(visualizer);
        }
        TestVisualizable other;
        other.setToggleKey(Qt::Key_G);
        other.bindToggle(visualizer);

        EXPECT_EQ(visualizer.shortcuts_created, 2);
        EXPECT_EQ(visualizer.panel_rows, (std::vector<std::string>{"Toggle Test visibility", "Toggle Test visibility"}));
        EXPECT_TRUE(visualizer.registry.allOn(Qt::Key_S));

        visualizer.registry.dispatch(Qt::Key_S);
        for (const auto &sensor : sensors)
            EXPECT_FALSE(sensor->isVisible());
        EXPECT_TRUE(other.isVisible());
        EXPECT_FALSE(visualizer.registry.allOn(Qt::Key_S));
        EXPECT_TRUE(visualizer.registry.allOn(Qt::Key_G));

        visualizer.registry.dispatch(Qt::Key_S);
        for (const auto &sensor : sensors)
            EXPECT_TRUE(sensor->isVisible());
        EXPECT_TRUE(visualizer.registry.allOn(Qt::Key_S));
    }

    TEST(KeyBindingRegistry, SharedKeyRowReadsOffWhileAnyMemberIsHidden)
    {
        RegistryBackedVisualizer visualizer;
        TestVisualizable a, b;
        a.setToggleKey(Qt::Key_S);
        b.setToggleKey(Qt::Key_S);
        a.bindToggle(visualizer);
        b.bindToggle(visualizer);

        b.toggleVisibility(); // out of sync with a
        EXPECT_FALSE(visualizer.registry.allOn(Qt::Key_S));
        visualizer.registry.dispatch(Qt::Key_S); // a -> hidden, b -> visible
        EXPECT_FALSE(a.isVisible());
        EXPECT_TRUE(b.isVisible());
        EXPECT_FALSE(visualizer.registry.allOn(Qt::Key_S));
    }
} // namespace Spacetime
