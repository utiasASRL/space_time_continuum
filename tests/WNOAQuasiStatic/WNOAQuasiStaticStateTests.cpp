#include <memory>
#include <stdexcept>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic.hpp>

namespace Spacetime::Priors::WNOAQuasiStatic
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Node<Pose>;
        using ValuesType = NodeValues<Pose>;
        using Keys = InterpolatedKeys<Pose>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;

        const Pose kPose(gtsam::Rot3::Rz(0.3), gtsam::Point3(0.1, 0.2, 0.3));
        const Vector6 kStrain = (Vector6() << 0.1, 0.0, -0.2, 0.0, 0.0, 1.0).finished();
    }

    TEST(WNOAQuasiStaticState, ValuesHoldOnlyAPoseAndAStrain)
    {
        EXPECT_EQ(ValuesType::dim, 12u);
        EXPECT_EQ(NodeType::dim, 12u);

        const ValuesType v(kPose, kStrain);
        EXPECT_EQ(v.values.size(), 2u);
        EXPECT_TRUE(v.pose().equals(kPose));
        EXPECT_TRUE(v.epsilon().isApprox(kStrain));

        const ValuesType identity;
        EXPECT_TRUE(identity.pose().equals(Pose()));
        EXPECT_TRUE(identity.epsilon().isApprox(Priors::straightStrain<Pose>())); // straight by default
    }

    TEST(WNOAQuasiStaticState, AssigningAnotherNodeTypesValuesThrows)
    {
        ValuesType v;
        Types::NodeValuesBase other;
        EXPECT_THROW(v = other, std::runtime_error);
    }

    TEST(WNOAQuasiStaticState, NodeRegistersAPoseAndAStrainVariable)
    {
        KeyManager km;
        const NodeType node(km, 0.5, 1.0);
        EXPECT_EQ(gtsam::Symbol(node.poseKey()).chr(), 'p');
        EXPECT_EQ(gtsam::Symbol(node.strainKey()).chr(), 's');
        EXPECT_EQ(node.keys().size(), 2u);

        const auto pairs = node.interpolatedKeys();
        ASSERT_EQ(pairs.size(), 2u);
        EXPECT_EQ(pairs[0], std::make_pair(Keys::Pose(), node.poseKey()));
        EXPECT_EQ(pairs[1], std::make_pair(Keys::Strain(), node.strainKey()));
    }

    TEST(WNOAQuasiStaticState, NodeValuesRoundTrip)
    {
        KeyManager km;
        const NodeType node(km, 0.0, 0.0);
        gtsam::Values values;
        node.setValues(values, ValuesType(kPose, kStrain));
        EXPECT_EQ(values.size(), 2u);

        const ValuesType read = node.getValues(values);
        EXPECT_TRUE(read.pose().equals(kPose));
        EXPECT_TRUE(read.epsilon().isApprox(kStrain));

        const auto base = std::dynamic_pointer_cast<ValuesType>(node.getValuesBase(values));
        ASSERT_TRUE(base);
        EXPECT_TRUE(base->pose().equals(kPose));
    }

    TEST(WNOAQuasiStaticState, InterpolatedKeysCoverThePoseAndStrainRows)
    {
        EXPECT_EQ(Keys::keys(), (gtsam::KeyVector{Keys::Pose(), Keys::Strain()}));
        EXPECT_EQ(Keys::rowRange(Keys::Pose()), std::make_pair(0, 6));
        EXPECT_EQ(Keys::rowRange(Keys::Strain()), std::make_pair(6, 6));
        EXPECT_THROW(Keys::rowRange(gtsam::Symbol('q', 2)), std::runtime_error);

        gtsam::Values values;
        Keys::insert(Keys::Strain(), ValuesType(kPose, kStrain), values);
        EXPECT_TRUE(values.at<Vector6>(Keys::Strain()).isApprox(kStrain));
        EXPECT_THROW(Keys::insert(gtsam::Symbol('q', 2), ValuesType(), values), std::runtime_error);
    }

} // namespace Spacetime::Priors::WNOAQuasiStatic
