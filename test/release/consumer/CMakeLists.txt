# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

cmake_minimum_required(VERSION 3.28)
project(WheelConsumer LANGUAGES CXX)
find_package(mqt-core CONFIG REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE MQT::CoreQDMI MQT::CoreDD Threads::Threads)
