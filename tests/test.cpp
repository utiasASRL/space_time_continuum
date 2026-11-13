#include <gtest/gtest.h>

int main(int argc, char **argv)
{
    srand(0);
    std::cout << "[INFO]: Setting random seed to 0" << std::endl;
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}