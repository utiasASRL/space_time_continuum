#include <functional>
#include <string>

#include <gtest/gtest.h>

#include <spacetime/visualizer/Visualizable.hpp>
#include <spacetime/visualizer/default/Visualizable.hpp>
#include <spacetime/visualizer/default/Visualizer.hpp>

namespace Spacetime
{
    namespace
    {
        /// Only the backend-agnostic base: no Qt/VTK involved.
        struct GenericVisualizable : public Visualization::Visualizable
        {
            explicit GenericVisualizable(bool is_static = false) : Visualization::Visualizable(is_static) {}
            bool isVisible() const { return visible_; }
            bool fresh(double display_time, double data_time) const { return isFresh(display_time, data_time); }
        };

        struct TestVisualizable : public Visualization::Default::Visualizable
        {
            explicit TestVisualizable(bool is_static = false) : Visualization::Default::Visualizable(is_static) {}
            bool isVisible() const { return visible_; }
            bool fresh(double display_time, double data_time) const { return isFresh(display_time, data_time); }

            template <typename VisualizerT>
            void bindToggle(VisualizerT &visualizer) const { bindVisibilityToggle(visualizer, "Test"); }
        };

        // Stands in for Visualization::Default::Visualizer's bindToggleShortcut (see
        // default/Visualizer.hpp) so bindVisibilityToggle's plumbing -- whether it wires
        // anything up at all, and what it wires up -- can be checked without a real
        // Qt/VTK Visualizer. Default::Visualizable::bindVisibilityToggle is templated
        // on the visualizer type specifically to allow this.
        struct FakeVisualizer
        {
            bool bound = false;
            Qt::Key bound_key = Qt::Key(0);
            std::string bound_label;
            std::function<void()> toggle_fn;
            std::function<bool()> state_fn;

            void bindToggleShortcut(Qt::Key key, std::function<void()> toggleFn, std::function<bool()> stateFn, const std::string &label)
            {
                bound = true;
                bound_key = key;
                bound_label = label;
                toggle_fn = std::move(toggleFn);
                state_fn = std::move(stateFn);
            }
        };
    }

    // Proves the base class and the IsVisualizable/IsVisualizableWithToggle
    // concepts (spacetime/visualizer/default/Visualizer.hpp) agree by construction --
    // this is the actual bug the Visualizable extraction fixes: previously
    // Robot/SensorBase each declared their own copy of these four methods,
    // with nothing checking they matched the concepts' expectations.
    static_assert(Visualization::Default::IsVisualizable<TestVisualizable>);
    static_assert(Visualization::Default::IsVisualizableWithToggle<TestVisualizable>);
    // The backend-agnostic base alone can't be drawn by the default backend: it has no visualize().
    static_assert(!Visualization::Default::IsVisualizable<GenericVisualizable>);

    TEST(Visualizable, ToggleVisibilityFlipsStateAndIsIdempotentOverTwoCalls)
    {
        GenericVisualizable v;
        v.toggleVisibility();
        EXPECT_FALSE(v.isVisible());
        v.toggleVisibility();
        EXPECT_TRUE(v.isVisible());
    }

    TEST(Visualizable, IsFreshWhileDataIsWithinTheStaleTimeoutOrFromTheFuture)
    {
        GenericVisualizable v;
        v.setStaleTimeout(1.0);
        EXPECT_TRUE(v.fresh(5.0, 5.0));
        EXPECT_TRUE(v.fresh(5.0, 6.0)); // data_time ahead of display_time
        EXPECT_TRUE(v.fresh(1.0, 0.0)); // exactly at the timeout
    }

    TEST(Visualizable, IsNotFreshOnceStaleTimeoutIsExceeded)
    {
        GenericVisualizable v;
        v.setStaleTimeout(1.0);
        EXPECT_FALSE(v.fresh(1.0001, 0.0));
        EXPECT_FALSE(v.fresh(10.0, 0.0));
    }

    TEST(Visualizable, StaticObjectsAreAlwaysFresh)
    {
        GenericVisualizable v(/*is_static=*/true);
        v.setStaleTimeout(1.0);
        EXPECT_TRUE(v.fresh(1000.0, 0.0));
    }

    TEST(Visualizable, BindVisibilityToggleIsANoOpWithoutAToggleKey)
    {
        TestVisualizable v;
        FakeVisualizer visualizer;
        v.bindToggle(visualizer);
        EXPECT_FALSE(visualizer.bound);
    }

    TEST(Visualizable, BindVisibilityToggleWiresTheConfiguredKeyAndLiveState)
    {
        TestVisualizable v;
        v.setToggleKey(Qt::Key_B);
        FakeVisualizer visualizer;
        v.bindToggle(visualizer);

        ASSERT_TRUE(visualizer.bound);
        EXPECT_EQ(visualizer.bound_key, Qt::Key_B);
        EXPECT_EQ(visualizer.bound_label, "Toggle Test visibility");

        EXPECT_TRUE(visualizer.state_fn());
        visualizer.toggle_fn();
        EXPECT_FALSE(visualizer.state_fn());
        EXPECT_FALSE(v.isVisible());
    }
}
