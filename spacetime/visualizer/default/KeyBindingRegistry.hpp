#ifndef KEY_BINDING_REGISTRY_HPP
#define KEY_BINDING_REGISTRY_HPP

#include <Qt>
#include <algorithm>
#include <functional>
#include <map>
#include <vector>

namespace Spacetime::Visualization::Default
{
    /**
     * @brief Per-key bookkeeping behind Visualizer::bindShortcut /
     *        bindToggleShortcut, so several objects can share one key (e.g.
     *        every PoseSensor bound to Qt::Key_S). Qt never emits
     *        QShortcut::activated for a key sequence claimed by more than one
     *        QShortcut in the same context (only activatedAmbiguously), so the
     *        Visualizer creates a single QShortcut per key and dispatches to
     *        every handler registered here. Kept free of any QObject/widget so
     *        it can be unit-tested without a QApplication.
     */
    class KeyBindingRegistry
    {
    public:
        /// Adds a handler for key. Returns true only for the key's first handler, i.e. when the caller should create the (single) QShortcut for it.
        bool addHandler(Qt::Key key, std::function<void()> fn)
        {
            auto &handlers = m_handlers[key];
            handlers.push_back(std::move(fn));
            return handlers.size() == 1;
        }

        /// Runs every handler registered for key, in registration order. A no-op for an unregistered key.
        void dispatch(Qt::Key key) const
        {
            auto it = m_handlers.find(key);
            if (it == m_handlers.end())
                return;
            for (const auto &fn : it->second)
                fn();
        }

        /// Adds a toggle-state query for key. Returns true only for the key's first one, i.e. when the caller should add the (single) diagnostics-panel row for it.
        bool addToggleState(Qt::Key key, std::function<bool()> stateFn)
        {
            auto &states = m_states[key];
            states.push_back(std::move(stateFn));
            return states.size() == 1;
        }

        /// Whether every toggle bound to key is on. Vacuously true for a key with no toggle states.
        bool allOn(Qt::Key key) const
        {
            auto it = m_states.find(key);
            if (it == m_states.end())
                return true;
            return std::all_of(it->second.begin(), it->second.end(), [](const auto &stateFn)
                               { return stateFn(); });
        }

    private:
        std::map<Qt::Key, std::vector<std::function<void()>>> m_handlers;
        std::map<Qt::Key, std::vector<std::function<bool()>>> m_states;
    };
} // namespace Spacetime::Visualization::Default

#endif // KEY_BINDING_REGISTRY_HPP
