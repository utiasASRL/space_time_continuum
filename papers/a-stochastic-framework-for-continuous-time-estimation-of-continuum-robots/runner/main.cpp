// TRO results runner: one batch estimate of one trial, written out for the result scripts.
//
// Usage:
//   tro_runner <config.json>... --output <dir>
//              [--quasi_static] [--prior_only] [--N <spatial nodes>] [--K <time nodes>] [--no_covariance]
//              [--time_samples <m>] [--shape_times <t1,t2,...>]
//
// --quasi_static runs the quasi-static benchmark instead: the WNOAQuasiStatic prior (with the
// arclength weights of prior.wnoa) and quasi-static estimation (a node row per measurement time).
// --prior_only solves a single node row at the start time without any measurement: the prior alone.
//
// Writes, in <dir>:
//   ground_truth/   the estimate at every ground-truth sample (error_pred.txt, error_pred_cov.txt)
//   tip_nodes/      the tip estimate at every node time
//   tip_samples/    the tip estimate at m evenly spaced times per node interval (--time_samples)
//   shape_<t>/      the estimate at 31 arclengths at time t (--shape_times, t relative to the start)
//   rows.txt        the time of every node row
//   runtime.json    solve time, covariance time, iterations, N and K
// Every estimate file has the format of Spacetime::Utilities::exportGroundTruthPredictions.

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtsam/geometry/Pose3.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/estimator/utilities/GroundTruthExport.hpp>

using Pose = gtsam::Pose3;
using WNOAPrior = Spacetime::Priors::WNOA::Prior<Pose>;
using QuasiStaticPrior = Spacetime::Priors::WNOAQuasiStatic::Prior<Pose>;

struct Arguments
{
    std::vector<std::string> config_files;
    std::string output;
    bool quasi_static = false;
    bool prior_only = false;
    int N = -1;                     // from the config if negative
    int K = -1;                     // from prior.time_period if negative
    bool no_covariance = false;
    int time_samples = 0;           // 0: no tip_samples output
    std::vector<double> shape_times;
};

Arguments parseArguments(int argc, char *argv[])
{
    Arguments args;
    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        if (arg == "--output")
            args.output = argv[++i];
        else if (arg == "--quasi_static")
            args.quasi_static = true;
        else if (arg == "--prior_only")
            args.prior_only = true;
        else if (arg == "--N")
            args.N = std::stoi(argv[++i]);
        else if (arg == "--K")
            args.K = std::stoi(argv[++i]);
        else if (arg == "--no_covariance")
            args.no_covariance = true;
        else if (arg == "--time_samples")
            args.time_samples = std::stoi(argv[++i]);
        else if (arg == "--shape_times")
        {
            std::stringstream list(argv[++i]);
            for (std::string t; std::getline(list, t, ',');)
                args.shape_times.push_back(std::stod(t));
        }
        else
            args.config_files.push_back(arg);
    }
    return args;
}

std::shared_ptr<WNOAPrior> makePrior(const Spacetime::Utilities::ConfigLoader &config, const std::shared_ptr<WNOAPrior::StateType> &state, double &time_period)
{
    Spacetime::Priors::WNOA::Values<Pose> base_start_point;
    WNOAPrior::Matrix3N Q0 = WNOAPrior::Matrix3N::Identity();
    WNOAPrior::MatrixN Q1 = WNOAPrior::MatrixN::Identity(), Q2 = Q1, Q3 = Q1, Q4 = Q1;
    config.loadWNOAPrior(base_start_point, time_period, Q0, Q1, Q2, Q3, Q4);
    return std::make_shared<WNOAPrior>(state, base_start_point, Q0, Q1, Q2, Q3, Q4, config.loadWNOAPriorOptions());
}

std::shared_ptr<QuasiStaticPrior> makePrior(const Spacetime::Utilities::ConfigLoader &config, const std::shared_ptr<QuasiStaticPrior::StateType> &state, double &time_period)
{
    QuasiStaticPrior::ValuesType base_start_point;
    QuasiStaticPrior::Matrix2N Q0 = QuasiStaticPrior::Matrix2N::Identity();
    QuasiStaticPrior::MatrixN Q2 = QuasiStaticPrior::MatrixN::Identity();
    config.loadWNOAQuasiStaticPrior<Pose>(base_start_point, time_period, Q0, Q2);
    return std::make_shared<QuasiStaticPrior>(state, base_start_point, Q0, Q2, config.loadWNOAQuasiStaticPriorOptions());
}

// Write the estimate at the given (time, arclength) points, by handing them to the ground-truth
// exporter as the samples of a placeholder sensor.
template <typename PriorT>
void exportAt(const std::shared_ptr<PriorT> &prior, const Spacetime::Estimator &estimator,
              const std::vector<std::pair<double, double>> &points, const std::string &directory)
{
    const Eigen::Matrix<double, 6, 6> unused_noise = Eigen::Matrix<double, 6, 6>::Identity();
    auto points_sensor = std::make_shared<Spacetime::Sensors::PoseSensor<typename PriorT::ValuesType>>(unused_noise, prior);
    for (const auto &[time, arclength] : points)
        points_sensor->addMeasurement(std::make_shared<Spacetime::Types::MeasurementBase>(
            arclength, time, Eigen::Matrix4d::Identity(), unused_noise, Eigen::MatrixXi::Ones(6, 1)));
    Spacetime::Utilities::exportGroundTruthPredictions(*prior, estimator.getCurrentValues(), estimator.getCurrentCovariance().get(),
                                                       {points_sensor}, directory);
}

template <typename PriorT>
int run(const Spacetime::Utilities::ConfigLoader &config, const Arguments &args)
{
    // Robot and estimator
    double length;
    unsigned int N;
    config.loadDemoMetadata(length, N);
    if (args.N > 0)
        N = args.N;
    auto state = std::make_shared<typename PriorT::StateType>(length, N);
    config.loadLocks(*state);

    Spacetime::EstimatorOptions options = config.loadEstimatorOptions();
    if (args.no_covariance)
        options.compute_covariance = false;
    options.quasi_static = args.quasi_static && !args.prior_only;
    Spacetime::Estimator estimator(options);

    double time_period = 1.0 / 30.0;
    auto prior = makePrior(config, state, time_period);
    estimator.addPrior(prior);

    std::vector<std::shared_ptr<Spacetime::Types::SensorBase>> sensors, ground_truth;
    config.loadDemoData(sensors, ground_truth, prior);
    if (!args.prior_only)
        for (const auto &sensor : sensors)
            estimator.addSensor(sensor);

    double start_time, end_time;
    config.loadDemoTimeWindow(start_time, end_time);

    // One estimate per node time, then the batch solve.
    int K = args.K > 0 ? args.K : static_cast<int>(std::round((end_time - start_time) / time_period)) + 1;
    if (args.prior_only)
        K = 1;
    time_period = K > 1 ? (end_time - start_time) / (K - 1) : 0.0;
    for (int k = 0; k < K; k++)
        estimator.estimate(start_time + k * time_period);
    estimator.finalize();

    // Outputs
    std::filesystem::create_directories(args.output);
    Spacetime::Utilities::exportGroundTruthPredictions(*prior, estimator.getCurrentValues(), estimator.getCurrentCovariance().get(),
                                                       ground_truth, args.output + "/ground_truth");

    std::vector<double> row_times;
    for (std::size_t i = 0; i < state->estimation_nodes.size(); i += N)
        row_times.push_back(state->estimation_nodes[i]->time);
    std::ofstream rows_file(args.output + "/rows.txt");
    rows_file.precision(17);
    for (const double t : row_times)
        rows_file << t << "\n";

    std::vector<std::pair<double, double>> tip_nodes;
    for (const double t : row_times)
        tip_nodes.push_back({t, length});
    exportAt(prior, estimator, tip_nodes, args.output + "/tip_nodes");

    if (args.time_samples > 0)
    {
        std::vector<std::pair<double, double>> tip_samples;
        for (std::size_t k = 0; k + 1 < row_times.size(); k++)
            for (int m = 0; m < args.time_samples; m++)
                tip_samples.push_back({row_times[k] + (row_times[k + 1] - row_times[k]) * m / args.time_samples, length});
        tip_samples.push_back({row_times.back(), length});
        exportAt(prior, estimator, tip_samples, args.output + "/tip_samples");
    }

    for (const double t : args.shape_times)
    {
        std::vector<std::pair<double, double>> shape;
        for (int i = 0; i <= 30; i++)
            shape.push_back({start_time + t, length * i / 30.0});
        std::ostringstream name;
        name << args.output << "/shape_" << t;
        exportAt(prior, estimator, shape, name.str());
    }

    const auto timings = estimator.getBatchTimings();
    std::ofstream runtime_file(args.output + "/runtime.json");
    runtime_file.precision(17);
    runtime_file << "{\n"
                 << "  \"solve\": " << timings.solve << ",\n"
                 << "  \"covariance\": " << timings.covariance << ",\n"
                 << "  \"iterations\": " << estimator.getBatchResult().iterations << ",\n"
                 << "  \"N\": " << N << ",\n"
                 << "  \"K\": " << row_times.size() << "\n"
                 << "}\n";
    std::cout << "Wrote " << args.output << " (" << row_times.size() << " rows, " << timings.solve << " s solve, "
              << estimator.getBatchResult().iterations << " iterations)" << std::endl;
    return 0;
}

int main(int argc, char *argv[])
{
    const Arguments args = parseArguments(argc, argv);
    if (args.config_files.empty() || args.output.empty())
    {
        std::cout << "Usage: " << argv[0] << " <config.json>... --output <dir> [--quasi_static] [--prior_only] [--N n] [--K k] [--no_covariance] "
                  << "[--time_samples m] [--shape_times t1,t2]" << std::endl;
        return 1;
    }

    Spacetime::Utilities::ConfigLoader config;
    for (const auto &file : args.config_files)
        if (!config.loadConfig(file))
            return 1;

    if (args.quasi_static)
        return run<QuasiStaticPrior>(config, args);
    return run<WNOAPrior>(config, args);
}
