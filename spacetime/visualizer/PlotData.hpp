#ifndef PLOTDATA_HPP
#define PLOTDATA_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <typeindex>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/types/MeasurementQueue.hpp>
#include <spacetime/estimator/types/States.hpp>
#include <spacetime/estimator/utilities/logger.hpp>

namespace Spacetime::Visualization
{
    /**
     * @brief Backend-agnostic data behind a state plot panel (see e.g.
     *        Default::PlotWidget): per-type plotters turning a node variable into
     *        scalar components with 1-sigma bounds, a history of every node's plotted
     *        values that outlives the estimator's fixed-lag window, slices of it along
     *        time or arclength, and ground-truth pose points. Free of any Qt/VTK
     *        dependency.
     */

    /// [x, y, z, roll, pitch, yaw] of `pose`: world position, then ZYX angles (R = Rz(yaw) Ry(pitch) Rx(roll), see gtsam::Rot3::rpy).
    inline std::array<double, 6> poseComponents(const gtsam::Pose3 &pose)
    {
        const gtsam::Vector3 rpy = pose.rotation().rpy();
        const gtsam::Point3 &t = pose.translation();
        return {t.x(), t.y(), t.z(), rpy(0), rpy(1), rpy(2)};
    }

    namespace detail
    {
        /// sqrt of a variance, clamped to be non-negative (a posterior can be very slightly indefinite numerically).
        inline double standardDeviation(double variance) { return std::sqrt(std::max(0.0, variance)); }

        /**
         * @brief d(roll, pitch, yaw) / d(omega) for the body-frame (right) rotation
         *        perturbation R * Exp(omega), with ZYX angles. nullopt near gimbal lock
         *        (|cos(pitch)| < 1e-6), where roll and yaw are not defined.
         */
        inline std::optional<Eigen::Matrix3d> eulerRateJacobian(const gtsam::Vector3 &rpy)
        {
            const double cp = std::cos(rpy(1));
            if (std::abs(cp) < 1e-6)
                return std::nullopt;
            const double sr = std::sin(rpy(0)), cr = std::cos(rpy(0)), tp = std::tan(rpy(1));
            Eigen::Matrix3d E;
            E << 1.0, sr * tp, cr * tp,
                0.0, cr, -sr,
                0.0, sr / cp, cr / cp;
            return E;
        }

        /// 1-sigma of each roll/pitch/yaw angle of `rotation`, from the 3 x 3 covariance of its body-frame perturbation (NaN at gimbal lock).
        inline std::array<double, 3> rpySigmas(const gtsam::Rot3 &rotation, const Eigen::Matrix3d &covariance)
        {
            constexpr double nan = std::numeric_limits<double>::quiet_NaN();
            const auto E = eulerRateJacobian(rotation.rpy());
            if (!E)
                return {nan, nan, nan};
            const Eigen::Matrix3d propagated = *E * covariance * E->transpose();
            return {standardDeviation(propagated(0, 0)), standardDeviation(propagated(1, 1)), standardDeviation(propagated(2, 2))};
        }
    } // namespace detail

    /**
     * @brief 1-sigma of each poseComponents() entry, propagated to first order from
     *        the 6 x 6 `covariance` of the pose's tangent in gtsam order
     *        [rotation; translation] with the right (body-frame) perturbation
     *        T = T_hat * Exp(delta). The position's world covariance is R Sigma_tt R^T
     *        (as for the robot view's covariance ellipsoids) and the angles' is
     *        E Sigma_rr E^T (see detail::eulerRateJacobian). All NaN if `covariance` is
     *        smaller than 6 x 6; the angles are NaN at gimbal lock.
     */
    inline std::array<double, 6> poseSigmas(const gtsam::Pose3 &pose, const Eigen::MatrixXd &covariance)
    {
        constexpr double nan = std::numeric_limits<double>::quiet_NaN();
        if (covariance.rows() < 6 || covariance.cols() < 6)
            return {nan, nan, nan, nan, nan, nan};
        const Eigen::Matrix3d R = pose.rotation().matrix();
        const Eigen::Matrix3d position = R * covariance.block<3, 3>(3, 3) * R.transpose();
        const auto angles = detail::rpySigmas(pose.rotation(), covariance.block<3, 3>(0, 0));
        return {detail::standardDeviation(position(0, 0)), detail::standardDeviation(position(1, 1)), detail::standardDeviation(position(2, 2)),
                angles[0], angles[1], angles[2]};
    }

    /// How one node-variable type is plotted: its component labels, values and 1-sigma bounds, read from a gtsam::Values entry.
    struct VariablePlotter
    {
        /// Labels of the components of the variable `name` stored at `key`.
        std::function<std::vector<std::string>(const std::string &name, const gtsam::Values &values, gtsam::Key key)> labels;
        /// Scalar components plotted for the variable stored at `key`.
        std::function<std::vector<double>(const gtsam::Values &values, gtsam::Key key)> components;
        /// 1-sigma of each component from the variable's marginal covariance (empty if unavailable, in which case every entry is NaN).
        std::function<std::vector<double>(const gtsam::Values &values, gtsam::Key key, const Eigen::MatrixXd &marginal)> sigmas;
    };

    /**
     * @brief Plotters by variable C++ type (Types::NodeBase::type). Constructed with the
     *        built-in ones: gtsam::Pose3 (x, y, z, roll, pitch, yaw; see poseComponents),
     *        gtsam::Pose2 (x, y, theta), gtsam::Rot3 (roll, pitch, yaw), double, Eigen
     *        vectors of size 1-6 and gtsam::Vector (one component each, in storage
     *        order). Other types can be added with registerPlotter.
     */
    class PlotterRegistry
    {
    public:
        PlotterRegistry()
        {
            registerPlotter<gtsam::Pose3>(
                [](const std::string &, const gtsam::Pose3 &)
                { return std::vector<std::string>{"x", "y", "z", "roll", "pitch", "yaw"}; },
                [](const gtsam::Pose3 &pose)
                { const auto c = poseComponents(pose); return std::vector<double>(c.begin(), c.end()); },
                [](const gtsam::Pose3 &pose, const Eigen::MatrixXd &marginal)
                { const auto s = poseSigmas(pose, marginal); return std::vector<double>(s.begin(), s.end()); });

            registerPlotter<gtsam::Pose2>(
                [](const std::string &, const gtsam::Pose2 &)
                { return std::vector<std::string>{"x", "y", "theta"}; },
                [](const gtsam::Pose2 &pose)
                { return std::vector<double>{pose.x(), pose.y(), pose.theta()}; },
                [](const gtsam::Pose2 &pose, const Eigen::MatrixXd &marginal)
                {
                    // Tangent [x; y; theta] with the right perturbation: world position covariance R Sigma_xy R^T.
                    if (marginal.rows() < 3 || marginal.cols() < 3)
                        return nanVector(3);
                    const Eigen::Matrix2d R = pose.rotation().matrix();
                    const Eigen::Matrix2d position = R * marginal.block<2, 2>(0, 0) * R.transpose();
                    return std::vector<double>{detail::standardDeviation(position(0, 0)), detail::standardDeviation(position(1, 1)),
                                               detail::standardDeviation(marginal(2, 2))};
                });

            registerPlotter<gtsam::Rot3>(
                [](const std::string &, const gtsam::Rot3 &)
                { return std::vector<std::string>{"roll", "pitch", "yaw"}; },
                [](const gtsam::Rot3 &rotation)
                { const gtsam::Vector3 rpy = rotation.rpy(); return std::vector<double>{rpy(0), rpy(1), rpy(2)}; },
                [](const gtsam::Rot3 &rotation, const Eigen::MatrixXd &marginal)
                {
                    if (marginal.rows() < 3 || marginal.cols() < 3)
                        return nanVector(3);
                    const auto s = detail::rpySigmas(rotation, marginal.block<3, 3>(0, 0));
                    return std::vector<double>(s.begin(), s.end());
                });

            registerPlotter<double>(
                [](const std::string &name, const double &)
                { return std::vector<std::string>{name}; },
                [](const double &value)
                { return std::vector<double>{value}; },
                [](const double &, const Eigen::MatrixXd &marginal)
                { return diagonalSigmas(marginal, 1); });

            registerVectorPlotter<gtsam::Vector1>();
            registerVectorPlotter<gtsam::Vector2>();
            registerVectorPlotter<gtsam::Vector3>();
            registerVectorPlotter<gtsam::Vector4>();
            registerVectorPlotter<gtsam::Vector5>();
            registerVectorPlotter<gtsam::Vector6>();
            registerVectorPlotter<gtsam::Vector>();
        }

        /**
         * @brief Adds (or replaces) the plotter for variables of type T.
         * @param labels     (variable name, value) -> one label per component.
         * @param components value -> the plotted scalar components.
         * @param sigmas     (value, marginal covariance in T's tangent space, empty if
         *                   unavailable) -> 1-sigma per component (NaN where unknown).
         */
        template <typename T>
        void registerPlotter(std::function<std::vector<std::string>(const std::string &, const T &)> labels,
                             std::function<std::vector<double>(const T &)> components,
                             std::function<std::vector<double>(const T &, const Eigen::MatrixXd &)> sigmas)
        {
            VariablePlotter plotter;
            plotter.labels = [labels = std::move(labels)](const std::string &name, const gtsam::Values &values, gtsam::Key key)
            { return labels(name, values.at<T>(key)); };
            plotter.components = [components = std::move(components)](const gtsam::Values &values, gtsam::Key key)
            { return components(values.at<T>(key)); };
            plotter.sigmas = [sigmas = std::move(sigmas)](const gtsam::Values &values, gtsam::Key key, const Eigen::MatrixXd &marginal)
            { return sigmas(values.at<T>(key), marginal); };
            m_plotters.insert_or_assign(std::type_index(typeid(T)), std::move(plotter));
        }

        /// The plotter for variables of `type`, or nullptr if none is registered.
        const VariablePlotter *find(std::type_index type) const
        {
            auto it = m_plotters.find(type);
            return it == m_plotters.end() ? nullptr : &it->second;
        }

    private:
        static std::vector<double> nanVector(std::size_t size) { return std::vector<double>(size, std::numeric_limits<double>::quiet_NaN()); }

        /// sqrt of the first `size` diagonal entries of `marginal`, or NaN if it is too small.
        static std::vector<double> diagonalSigmas(const Eigen::MatrixXd &marginal, std::size_t size)
        {
            if (marginal.rows() < static_cast<Eigen::Index>(size) || marginal.cols() < static_cast<Eigen::Index>(size))
                return nanVector(size);
            std::vector<double> out(size);
            for (std::size_t i = 0; i < size; i++)
                out[i] = detail::standardDeviation(marginal(i, i));
            return out;
        }

        /// One component per entry, labelled `name[i]`, with sigma = sqrt(diag(marginal)).
        template <typename VectorT>
        void registerVectorPlotter()
        {
            registerPlotter<VectorT>(
                [](const std::string &name, const VectorT &value)
                {
                    std::vector<std::string> out;
                    for (Eigen::Index i = 0; i < value.size(); i++)
                        out.push_back(name + "[" + std::to_string(i) + "]");
                    return out;
                },
                [](const VectorT &value)
                { return std::vector<double>(value.data(), value.data() + value.size()); },
                [](const VectorT &value, const Eigen::MatrixXd &marginal)
                { return diagonalSigmas(marginal, static_cast<std::size_t>(value.size())); });
        }

        std::map<std::type_index, VariablePlotter> m_plotters;
    };

    /// A plotted line: the mean at each x, with mean -/+ 3 sigma bounds (NaN where no covariance is available).
    struct PlotSeries
    {
        std::vector<double> x, mean, lower, upper;
    };

    /// The data window a plot shows.
    struct PlotRange
    {
        double x_min, x_max, y_min, y_max;
    };

    /**
     * @brief The range a plot of `series` (mean and finite bounds) and ground truth
     *        (`gt_x`, `gt_y`) is fitted to: every finite value, with the value range
     *        padded by 10% each side. Degenerate ranges (one sample, a constant value)
     *        are widened. nullopt if nothing is finite.
     */
    inline std::optional<PlotRange> fitRange(const PlotSeries &series, const std::vector<double> &gt_x, const std::vector<double> &gt_y)
    {
        double x_min = std::numeric_limits<double>::infinity(), x_max = -x_min;
        double y_min = x_min, y_max = -x_min;
        auto include = [&](const std::vector<double> &xs, const std::vector<double> &ys)
        {
            if (xs.size() != ys.size())
                return;
            for (std::size_t i = 0; i < xs.size(); i++)
                if (std::isfinite(xs[i]) && std::isfinite(ys[i]))
                {
                    x_min = std::min(x_min, xs[i]);
                    x_max = std::max(x_max, xs[i]);
                    y_min = std::min(y_min, ys[i]);
                    y_max = std::max(y_max, ys[i]);
                }
        };
        include(series.x, series.mean);
        if (series.mean.size() == series.x.size())
        {
            include(series.x, series.lower);
            include(series.x, series.upper);
        }
        include(gt_x, gt_y);
        if (!std::isfinite(x_min))
            return std::nullopt;

        if (x_max - x_min < 1e-12)
        {
            x_min -= 0.5;
            x_max += 0.5;
        }
        double y_span = y_max - y_min;
        if (y_span < 1e-12)
            y_span = std::max(1.0, std::abs(y_min));
        return PlotRange{x_min, x_max, y_min - 0.1 * y_span, y_max + 0.1 * y_span};
    }

    /**
     * @brief `range` zoomed about (anchor_x, anchor_y): each axis' span is multiplied
     *        by its scale (< 1 zooms in, 1 leaves the axis unchanged), keeping the
     *        anchor at the same place on screen. Spans are kept within [1e-12, 1e12].
     */
    inline PlotRange zoomRange(const PlotRange &range, double anchor_x, double anchor_y, double x_scale, double y_scale)
    {
        auto zoom = [](double lo, double hi, double anchor, double scale)
        {
            const double span = hi - lo;
            const double new_span = std::clamp(span * scale, 1e-12, 1e12);
            const double fraction = (anchor - lo) / span; // where the anchor sits, kept fixed
            const double new_lo = anchor - fraction * new_span;
            return std::pair{new_lo, new_lo + new_span};
        };
        PlotRange out = range;
        if (x_scale != 1.0)
            std::tie(out.x_min, out.x_max) = zoom(range.x_min, range.x_max, anchor_x, x_scale);
        if (y_scale != 1.0)
            std::tie(out.y_min, out.y_max) = zoom(range.y_min, range.y_max, anchor_y, y_scale);
        return out;
    }

    /// `range` shifted by (dx, dy) data units.
    inline PlotRange panRange(const PlotRange &range, double dx, double dy)
    {
        return PlotRange{range.x_min + dx, range.x_max + dx, range.y_min + dy, range.y_max + dy};
    }

    /// The plots of one node variable: one series per component. `supported` is false (and `series` empty) for a type with no registered plotter.
    struct PlotGroup
    {
        std::string name;
        std::vector<std::string> labels;
        bool supported = true;
        std::vector<PlotSeries> series;
        std::type_index type = typeid(void);
    };

    /// How one node's variable is plotted: its name, type, component labels and whether a plotter exists for it.
    struct VariableLayout
    {
        std::string name;
        std::type_index type = typeid(void);
        std::vector<std::string> labels;
        bool supported = true;

        bool operator==(const VariableLayout &other) const = default;
    };
    using NodeLayout = std::vector<VariableLayout>;

    /**
     * @brief The plotted values of every node seen so far, one entry per flat node
     *        index (row-major: row = time, column = arclength, as
     *        Types::ContinuumRobotState lays them out). Each ingest() refreshes the
     *        rows present in the snapshot; rows that have since left it (e.g.
     *        marginalized by a fixed-lag smoother) keep their last value, so the whole
     *        run stays plottable. Which variables are plotted follows each node's own
     *        variables (Types::NodeBase::names) -- nothing is specific to a prior.
     *        Not thread-safe: use from one (the visualizer's) thread.
     */
    class NodeHistory
    {
    public:
        /// One node's cached values.
        struct Entry
        {
            bool valid = false;
            double time = 0.0, arclength = 0.0;
            std::shared_ptr<const NodeLayout> layout;
            /// Per variable (in layout order): components and their 1-sigma.
            std::vector<std::vector<double>> mean, sigma;
        };

        /// Plotters used for subsequent ingests; register custom variable types here.
        PlotterRegistry &plotters() { return m_plotters; }

        /**
         * @brief Refreshes the entries of every row of `nodes` whose variables are all in
         *        `snapshot.values`. Rows are walked newest first: rows not in the snapshot
         *        yet (e.g. mid-solve, or a historical snapshot after rewinding) are
         *        skipped, and the walk stops at the first missing row older than one that
         *        was found (marginalized rows keep their cached value).
         * @param nodes       Every node of the state, row-major (see ContinuumRobotState::snapshotEstimationNodes).
         * @param spatial_dim Nodes per row.
         * @return Whether any entry was refreshed.
         */
        template <Types::NodeType NodeT>
        bool ingest(const std::vector<std::shared_ptr<NodeT>> &nodes, unsigned int spatial_dim, const EstimatorSnapshot &snapshot)
        {
            if (spatial_dim == 0)
                return false;
            m_spatial_dim = spatial_dim;
            const std::size_t num_rows = nodes.size() / spatial_dim;

            bool found_row = false;
            for (std::size_t row = num_rows; row-- > 0;)
            {
                bool present = true;
                for (unsigned int i = 0; i < spatial_dim && present; i++)
                    for (const gtsam::Key key : nodes[row * spatial_dim + i]->keys())
                        if (!snapshot.values.exists(key))
                        {
                            present = false;
                            break;
                        }
                if (!present)
                {
                    if (found_row)
                        break;
                    continue;
                }
                if (!found_row && m_entries.size() < (row + 1) * spatial_dim)
                    m_entries.resize((row + 1) * spatial_dim);
                found_row = true;
                for (unsigned int i = 0; i < spatial_dim; i++)
                    m_entries[row * spatial_dim + i] = makeEntry(*nodes[row * spatial_dim + i], snapshot);
            }
            if (found_row)
                m_revision++;
            return found_row;
        }

        /// Nodes per row (0 before the first ingest).
        unsigned int spatialDim() const { return m_spatial_dim; }
        /// Rows with an entry slot (the newest ingested row + 1).
        std::size_t numRows() const { return m_spatial_dim ? m_entries.size() / m_spatial_dim : 0; }
        /// Incremented by every ingest that refreshed an entry.
        unsigned long revision() const { return m_revision; }

        /// The cached entry at (row, spatial), or nullptr if out of range or never ingested.
        const Entry *entry(std::size_t row, unsigned int spatial) const
        {
            if (spatial >= m_spatial_dim || row >= numRows())
                return nullptr;
            const Entry &e = m_entries[row * m_spatial_dim + spatial];
            return e.valid ? &e : nullptr;
        }

        /// Time of `row` (its first cached node's), or nullopt if it has no entry.
        std::optional<double> rowTime(std::size_t row) const
        {
            for (unsigned int i = 0; i < m_spatial_dim; i++)
                if (const Entry *e = entry(row, i))
                    return e->time;
            return std::nullopt;
        }

        /// The newest cached row with time <= `time`, or nullopt if there is none.
        std::optional<std::size_t> latestRowAtOrBefore(double time) const
        {
            for (std::size_t row = numRows(); row-- > 0;)
                if (const auto t = rowTime(row); t && *t <= time)
                    return row;
            return std::nullopt;
        }

        /// Every variable of spatial node `spatial` against time, over the cached rows with time <= `max_time`. Laid out as the newest such node.
        std::vector<PlotGroup> sliceAtArclength(unsigned int spatial, double max_time) const
        {
            std::vector<const Entry *> entries;
            for (std::size_t row = 0; row < numRows(); row++)
                if (const Entry *e = entry(row, spatial); e && e->time <= max_time)
                    entries.push_back(e);
            return slice(entries, [](const Entry &e)
                         { return e.time; });
        }

        /// Every variable of row `row` against arclength. Laid out as its last cached node.
        std::vector<PlotGroup> sliceAtTime(std::size_t row) const
        {
            std::vector<const Entry *> entries;
            for (unsigned int i = 0; i < m_spatial_dim; i++)
                if (const Entry *e = entry(row, i))
                    entries.push_back(e);
            return slice(entries, [](const Entry &e)
                         { return e.arclength; });
        }

    private:
        /// Reads every variable of `node` from `snapshot` through its type's plotter.
        Entry makeEntry(const Types::NodeBase &node, const EstimatorSnapshot &snapshot)
        {
            Entry entry;
            entry.valid = true;
            entry.time = node.time;
            entry.arclength = node.arclength;

            NodeLayout layout;
            for (const std::string &name : node.names())
            {
                const gtsam::Key key = node.key(name);
                VariableLayout variable{name, node.type(name), {}, true};
                const VariablePlotter *plotter = m_plotters.find(variable.type);
                if (!plotter)
                {
                    variable.supported = false;
                    if (m_warned_types.insert(variable.type).second)
                        warning() << "NodeHistory: no plotter registered for variable '" << name << "' of type " << variable.type.name()
                                  << "; it will not be plotted (see PlotterRegistry::registerPlotter).";
                    entry.mean.emplace_back();
                    entry.sigma.emplace_back();
                }
                else
                {
                    variable.labels = plotter->labels(name, snapshot.values, key);
                    entry.mean.push_back(plotter->components(snapshot.values, key));
                    const bool has_marginal = snapshot.covariance && snapshot.covariance->has(key);
                    entry.sigma.push_back(plotter->sigmas(snapshot.values, key, has_marginal ? Eigen::MatrixXd(snapshot.covariance->covariance(key)) : Eigen::MatrixXd()));
                }
                layout.push_back(std::move(variable));
            }

            // Nodes of one type share a layout: reuse the last one so slices compare layouts by pointer.
            if (!m_last_layout || *m_last_layout != layout)
                m_last_layout = std::make_shared<const NodeLayout>(std::move(layout));
            entry.layout = m_last_layout;
            return entry;
        }

        /// Groups for `entries` (in order), laid out as the last one; entries with a different layout are skipped.
        template <typename XFn>
        static std::vector<PlotGroup> slice(const std::vector<const Entry *> &entries, XFn x_of)
        {
            if (entries.empty())
                return {};
            const auto &reference = entries.back()->layout;
            std::vector<PlotGroup> groups;
            for (const VariableLayout &variable : *reference)
            {
                PlotGroup group{variable.name, variable.labels, variable.supported, {}, variable.type};
                if (variable.supported)
                    group.series.resize(variable.labels.size());
                groups.push_back(std::move(group));
            }

            for (const Entry *e : entries)
            {
                if (e->layout != reference && *e->layout != *reference)
                    continue;
                const double x = x_of(*e);
                for (std::size_t v = 0; v < groups.size(); v++)
                    for (std::size_t c = 0; c < groups[v].series.size(); c++)
                    {
                        const double mean = e->mean[v][c];
                        const double sigma = c < e->sigma[v].size() ? e->sigma[v][c] : std::numeric_limits<double>::quiet_NaN();
                        PlotSeries &series = groups[v].series[c];
                        series.x.push_back(x);
                        series.mean.push_back(mean);
                        series.lower.push_back(mean - 3.0 * sigma);
                        series.upper.push_back(mean + 3.0 * sigma);
                    }
            }
            return groups;
        }

        PlotterRegistry m_plotters;
        std::vector<Entry> m_entries;
        unsigned int m_spatial_dim = 0;
        unsigned long m_revision = 0;
        std::shared_ptr<const NodeLayout> m_last_layout;
        std::set<std::type_index> m_warned_types;
    };

    /// Ground-truth points: x (time or arclength) and, per poseComponents() entry, the value at each x.
    struct GroundTruthPoints
    {
        std::vector<double> x;
        std::vector<std::vector<double>> components = std::vector<std::vector<double>>(6);
    };

    namespace detail
    {
        /// Appends the poseComponents() of `measurement` (a 4 x 4 pose; other shapes are skipped) at `x`.
        inline void appendPose(GroundTruthPoints &points, double x, const Types::MeasurementBase &measurement)
        {
            if (measurement.value.rows() != 4 || measurement.value.cols() != 4)
                return;
            const auto c = poseComponents(gtsam::Pose3(Eigen::Matrix4d(measurement.value)));
            points.x.push_back(x);
            for (std::size_t i = 0; i < 6; i++)
                points.components[i].push_back(c[i]);
        }

        /// `points` reordered by ascending x.
        inline GroundTruthPoints sortedByX(const GroundTruthPoints &points)
        {
            std::vector<std::size_t> order(points.x.size());
            for (std::size_t i = 0; i < order.size(); i++)
                order[i] = i;
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b)
                             { return points.x[a] < points.x[b]; });
            GroundTruthPoints out;
            for (std::size_t i : order)
            {
                out.x.push_back(points.x[i]);
                for (std::size_t c = 0; c < 6; c++)
                    out.components[c].push_back(points.components[c][i]);
            }
            return out;
        }
    } // namespace detail

    /**
     * @brief Pose ground truth against time at `arclength`: the measurements with
     *        time <= `max_time` of the one sensor (each sensor's measurements are
     *        assumed to share an arclength) closest to `arclength`, if within
     *        `tolerance`. Ordered by time.
     */
    inline GroundTruthPoints poseGroundTruthAtArclength(const std::vector<std::vector<Types::MeasurementQueue::pMeasurementType>> &sensors,
                                                        double arclength, double tolerance, double max_time)
    {
        const std::vector<Types::MeasurementQueue::pMeasurementType> *closest = nullptr;
        double closest_distance = tolerance;
        for (const auto &measurements : sensors)
        {
            if (measurements.empty())
                continue;
            const double distance = std::abs(measurements.front()->arclength - arclength);
            if (distance <= closest_distance)
            {
                closest = &measurements;
                closest_distance = distance;
            }
        }

        GroundTruthPoints points;
        if (closest)
            for (const auto &measurement : *closest)
                if (measurement->time <= max_time)
                    detail::appendPose(points, measurement->time, *measurement);
        return detail::sortedByX(points);
    }

    /// Pose ground truth against arclength at `time`: each sensor's measurement nearest `time`, if within `tolerance`. Ordered by arclength.
    inline GroundTruthPoints poseGroundTruthAtTime(const std::vector<std::vector<Types::MeasurementQueue::pMeasurementType>> &sensors,
                                                   double time, double tolerance)
    {
        GroundTruthPoints points;
        for (const auto &measurements : sensors)
        {
            const Types::MeasurementBase *nearest = nullptr;
            double nearest_distance = tolerance;
            for (const auto &measurement : measurements)
            {
                const double distance = std::abs(measurement->time - time);
                if (distance <= nearest_distance)
                {
                    nearest = measurement.get();
                    nearest_distance = distance;
                }
            }
            if (nearest)
                detail::appendPose(points, nearest->arclength, *nearest);
        }
        return detail::sortedByX(points);
    }

    /// Which way a plot panel slices the state.
    enum class PlotMode
    {
        TimeAtArclength, ///< x-axis = time, at the selected node's arclength.
        ArclengthAtTime  ///< x-axis = arclength, at the displayed time.
    };

    /// The plot panel's selected spatial node and mode. Starts at the tip, in TimeAtArclength mode.
    class PlotSelection
    {
    public:
        explicit PlotSelection(unsigned int spatial_dim)
            : m_spatial_dim(spatial_dim), m_spatial_index(spatial_dim ? spatial_dim - 1 : 0) {}

        unsigned int spatialIndex() const { return m_spatial_index; }
        PlotMode mode() const { return m_mode; }
        /// Incremented by every change of selection or mode.
        unsigned long revision() const { return m_revision; }

        /// Moves the selection by `delta` nodes, clamped to the robot. Returns whether it changed.
        bool step(int delta)
        {
            if (m_spatial_dim == 0)
                return false;
            const int target = std::clamp(static_cast<int>(m_spatial_index) + delta, 0, static_cast<int>(m_spatial_dim) - 1);
            return select(static_cast<unsigned int>(target));
        }

        /// Selects node `spatial_index`. Returns false (and changes nothing) if it is out of range or already selected.
        bool select(unsigned int spatial_index)
        {
            if (spatial_index >= m_spatial_dim || spatial_index == m_spatial_index)
                return false;
            m_spatial_index = spatial_index;
            m_revision++;
            return true;
        }

        /// Switches between TimeAtArclength and ArclengthAtTime.
        void toggleMode()
        {
            m_mode = (m_mode == PlotMode::TimeAtArclength) ? PlotMode::ArclengthAtTime : PlotMode::TimeAtArclength;
            m_revision++;
        }

    private:
        unsigned int m_spatial_dim;
        unsigned int m_spatial_index;
        PlotMode m_mode = PlotMode::TimeAtArclength;
        unsigned long m_revision = 0;
    };
} // namespace Spacetime::Visualization

#endif // PLOTDATA_HPP
