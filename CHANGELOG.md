# Change Log

All notable changes to this repository will be documented here.

## [Unreleased]

Features that I am currently working on including in this library and can be expected within the next few months include:

- More factor types 
- Tutorials to better explain how to use the library in your own projects

## Nov 13, 2026

This update marks a significant change to the project that is certain to break legacy code. The changes included aim to improve the project and move it into "practically useful" territory. This commit is consistent with the results published in "A Stochastic Framework for Continuous-Time State Estimation of Continuum Robots" at IEEE Transactions on Robotics. These changes include:

- The project has been switched to the GTSAM library to take advantage of established tools for factor-graph optimization. This results in an approximate 10x improvement in runtime across all applications.
- An interface abstraction has been introduced, providing options for both on and offline use. This includes a ROS2 interface online estimation of real robots.
- Several additional factors have been added, including joining, actuation, and accelerometer ones. Additionally, biases can now be estimated in the state.
- Python bindings have been added!
- A number of examples have been added for both the C++ and Python implementations

## February 19, 2026

- Initial commit of project code. Added core functionality and demos associated with [1] and [2].
