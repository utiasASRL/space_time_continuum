#ifndef VISUALIZABLE_HPP
#define VISUALIZABLE_HPP

#include <string>

namespace Spacetime::Visualization
{
    /**
     * @brief Backend-agnostic base for anything a visualizer backend can display:
     *        visibility and staleness bookkeeping plus the per-frame update hook.
     *        Deliberately free of any Qt/VTK (or other backend) dependency -- each
     *        backend derives its own base from this one to add how an object is
     *        first drawn and how its controls are bound (see e.g.
     *        Default::Visualizable in spacetime/visualizer/default/Visualizable.hpp).
     */
    struct Visualizable
    {
        /// @param is_static Whether this object has no data source and so is never considered stale (see isFresh).
        explicit Visualizable(bool is_static = false) : is_static_(is_static) {}
        /// Virtual destructor for polymorphic deletion.
        virtual ~Visualizable() = default;

        /// Human-readable description: a short name if `brief`, otherwise a longer summary. Empty by default.
        virtual std::string info(bool brief = true) const { return std::string(); }
        /**
         * @brief Per-frame update hook, called by the visualizer on every rendered frame. No-op by default.
         * @param time          Time to draw this object at (playback time minus the
         *                      visualizer's phase delay).
         * @param estimate_time Current playback time, i.e. which estimator snapshot
         *                      to read from. Equal to `time` when there is no delay.
         */
        virtual void updateVisualization(double time, double estimate_time) const {}
        /// Flips visible_. Subclasses owning drawn objects should override to also apply it to them.
        virtual void toggleVisibility() const { visible_ = !visible_; }

        /// Seconds after its last data before a non-static object is hidden as stale (see isFresh).
        void setStaleTimeout(double stale_timeout) { stale_timeout_ = stale_timeout; }

    protected:
        /**
         * @brief Whether something last backed by data at data_time should still
         *        be shown at display_time: true as long as display_time is within
         *        stale_timeout_ seconds of data_time. is_static_ objects (e.g.
         *        scene decorations with no data source) are always fresh.
         */
        bool isFresh(double display_time, double data_time) const
        {
            if (is_static_)
                return true;
            return (display_time - data_time) <= stale_timeout_;
        }

        /// User-toggled visibility (see toggleVisibility).
        mutable bool visible_ = true;
        /// Never stale when true (see isFresh).
        bool is_static_ = false;
        /// [s], see setStaleTimeout.
        double stale_timeout_ = 0.2;
    };
} // namespace Spacetime::Visualization

#endif // VISUALIZABLE_HPP
