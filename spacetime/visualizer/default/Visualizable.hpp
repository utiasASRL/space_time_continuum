#ifndef DEFAULT_VISUALIZABLE_HPP
#define DEFAULT_VISUALIZABLE_HPP

#include <Qt>
#include <optional>
#include <string>

#include <spacetime/visualizer/Visualizable.hpp>

namespace Spacetime::Visualization::Default
{
    class Visualizer;

    /**
     * @brief Base for every object view of the default (Qt/VTK) backend: the
     *        backend-agnostic Visualization::Visualizable plus how the object is
     *        first drawn into a Default::Visualizer and which key toggles it.
     *        Designed to structurally satisfy IsVisualizable /
     *        IsVisualizableWithToggle (spacetime/visualizer/default/Visualizer.hpp).
     */
    struct Visualizable : public Visualization::Visualizable
    {
        using Visualization::Visualizable::Visualizable;

        /// Creates this object's actors in `visualizer` and binds its controls. Called once, synchronously, by Visualizer::addVisualizable. No-op by default.
        virtual void visualize(Visualizer &visualizer) const {}

        /**
         * @brief Sets the key a subclass's visualize() binds (via
         *        bindVisibilityToggle) to toggle this object's visibility, so
         *        instances of one type can use different keys (e.g. a regular
         *        vs. a ground-truth sensor). Must be called before
         *        Visualizer::addVisualizable, which calls visualize() synchronously.
         */
        void setToggleKey(Qt::Key key) { toggle_key_ = key; }

        /// Sets whether this object starts shown (true by default); its toggle key flips it from there. Call before Visualizer::addVisualizable.
        void setVisible(bool visible) { visible_ = visible; }

    protected:
        /**
         * @brief If setToggleKey() was called, binds that key to toggleVisibility()
         *        plus a diagnostics-panel row showing the live 1/0 state, via
         *        Visualizer::bindToggleShortcut. A no-op otherwise. Call it from a
         *        subclass's visualize() override.
         *
         *        Templated on the visualizer type only to avoid a circular include
         *        (Visualizer.hpp includes this header).
         * @param name Object name used in the row's label ("Toggle <name> visibility").
         */
        template <typename VisualizerT>
        void bindVisibilityToggle(VisualizerT &visualizer, const std::string &name) const
        {
            if (!toggle_key_)
                return;
            visualizer.bindToggleShortcut(*toggle_key_, [this]()
                                          { toggleVisibility(); }, [this]()
                                          { return visible_; }, "Toggle " + name + " visibility");
        }

        /// See setToggleKey.
        std::optional<Qt::Key> toggle_key_;
    };
} // namespace Spacetime::Visualization::Default

#endif // DEFAULT_VISUALIZABLE_HPP
