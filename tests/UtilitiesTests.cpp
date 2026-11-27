#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <json/json.h>

#include <spacetime/estimator/utilities/functions.hpp>
#include <spacetime/estimator/utilities/logger.hpp>

namespace
{
    /// A file under the test's own temporary folder (ctest runs tests concurrently).
    std::string tempFile(const std::string &name, const std::string &contents)
    {
        const std::filesystem::path dir = std::filesystem::path(testing::TempDir()) /
                                          (std::string("utilities_") + testing::UnitTest::GetInstance()->current_test_info()->name());
        std::filesystem::create_directories(dir);
        std::ofstream(dir / name) << contents;
        return (dir / name).string();
    }

    Json::Value parse(const std::string &text)
    {
        Json::Value root;
        std::istringstream(text) >> root;
        return root;
    }

    /// Points std::cout at its own buffer for the test's lifetime, restoring it afterwards.
    struct CoutCapture
    {
        std::ostringstream out;
        std::streambuf *previous = std::cout.rdbuf(out.rdbuf());
        ~CoutCapture() { std::cout.rdbuf(previous); }
    };
} // namespace

// ---- logger capture ---------------------------------------------------------------------

TEST(LoggerCapture, StartingASecondCaptureEndsTheFirst)
{
    CoutCapture console;
    infoStart();
    std::cout << "first";
    warningStart(); // used to leave std::cout pointing at the first capture's freed buffer
    std::cout << "second";
    logReset();
    std::cout << "after";

    const std::string text = console.out.str();
    EXPECT_NE(text.find("first"), std::string::npos);
    EXPECT_NE(text.find("second"), std::string::npos);
    EXPECT_NE(text.find("after"), std::string::npos);
    EXPECT_LT(text.find("first"), text.find("second"));
}

TEST(LoggerCapture, DestroyingACaptureRestoresStdCout)
{
    CoutCapture console;
    {
        RedirectLogger capture(LogLevel::Info);
        std::cout << "captured";
    } // no logReset()
    std::cout << "after";
    EXPECT_NE(console.out.str().find("captured"), std::string::npos);
    EXPECT_NE(console.out.str().find("after"), std::string::npos);
}

// ---- loadCsv ----------------------------------------------------------------------------

TEST(LoadCsv, ReadsRowsAndSkipsBlankLines)
{
    const auto m = loadCsv<Eigen::MatrixXd>(tempFile("a.csv", "1,2,3\n4,5,6\n\n"));
    ASSERT_EQ(m.rows(), 2);
    ASSERT_EQ(m.cols(), 3);
    EXPECT_EQ(m(1, 2), 6.0);
}

TEST(LoadCsv, RejectsMissingEmptyRaggedAndMisfittingFiles)
{
    EXPECT_THROW(loadCsv<Eigen::MatrixXd>(tempFile("unused", "") + ".missing"), std::runtime_error);
    EXPECT_THROW(loadCsv<Eigen::MatrixXd>(tempFile("empty.csv", "")), std::runtime_error);
    EXPECT_THROW(loadCsv<Eigen::MatrixXd>(tempFile("ragged.csv", "1,2\n3\n")), std::runtime_error);
    EXPECT_THROW((loadCsv<Eigen::Matrix<double, 2, 2>>(tempFile("big.csv", "1,2,3\n4,5,6\n"))), std::runtime_error);
}

// ---- JSON arrays ------------------------------------------------------------------------

TEST(LoadArrayData, FixedSizeMustMatchTheArrayLength)
{
    Json::Value root = parse(R"({ "ok": [1, 2, 3], "short": [1, 2], "long": [1, 2, 3, 4], "text": "abc" })");
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    EXPECT_TRUE(loadArrayData(root, "ok", v));
    EXPECT_EQ(v, Eigen::Vector3d(1, 2, 3));
    EXPECT_THROW(loadArrayData(root, "short", v), std::invalid_argument);
    EXPECT_THROW(loadArrayData(root, "long", v), std::invalid_argument);
    EXPECT_THROW(loadArrayData(root, "text", v), std::invalid_argument);
    EXPECT_FALSE(loadArrayData(root, "absent", v));
    EXPECT_EQ(v, Eigen::Vector3d(1, 2, 3)); // unchanged by the failures
}

TEST(LoadArrayData, DynamicSizesAreFittedToTheArray)
{
    Json::Value root = parse(R"({ "six": [1, 2, 3, 4, 5, 6], "five": [1, 2, 3, 4, 5] })");
    Eigen::VectorXd v;
    EXPECT_TRUE(loadArrayData(root, "six", v));
    EXPECT_EQ(v.size(), 6);
    Eigen::Matrix<double, 2, Eigen::Dynamic> m;
    EXPECT_TRUE(loadArrayData(root, "six", m));
    EXPECT_EQ(m.cols(), 3);
    EXPECT_THROW(loadArrayData(root, "five", m), std::invalid_argument);
}

TEST(LoadMatrix4Data, NeedsExactlySixteenNumbers)
{
    Json::Value root = parse(R"({ "T": [1, 0, 0, 5, 0, 1, 0, 6, 0, 0, 1, 7, 0, 0, 0, 1], "short": [1, 0, 0, 5] })");
    Eigen::Matrix4d T = Eigen::Matrix4d::Zero();
    EXPECT_TRUE(loadMatrix4Data(root, "T", T));
    EXPECT_EQ(T(0, 3), 5.0); // row-major in the file
    EXPECT_EQ(T(2, 3), 7.0);
    EXPECT_THROW(loadMatrix4Data(root, "short", T), std::invalid_argument);
}
