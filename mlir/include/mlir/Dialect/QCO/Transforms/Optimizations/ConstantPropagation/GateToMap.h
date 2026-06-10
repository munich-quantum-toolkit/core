/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_GATETOMAP_H
#define MQT_CORE_GATETOMAP_H

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Drivers.h"

#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using ResultMap =
    std::unordered_map<unsigned int,
                       std::unordered_map<unsigned int, std::complex<double>>>;

inline std::unordered_map<
    unsigned int, std::unordered_map<unsigned int, std::complex<double>>>
getQubitMappingOfGates(mlir::Operation* gate,
                       const std::vector<double>& params) {

  return mlir::TypeSwitch<
             mlir::Operation*,
             std::unordered_map<
                 unsigned int,
                 std::unordered_map<unsigned int, std::complex<double>>>>(gate)
      .Case<mlir::qco::IdOp>([&](auto) {
        return ResultMap{{0, {{0, std::complex<double>(1, 0)}}},
                         {1, {{1, std::complex<double>(1, 0)}}}};
      })
      .Case<mlir::qco::HOp>([&](auto) {
        return ResultMap{
            {{0,
              {{0, std::complex<double>(1 / std::numbers::sqrt2, 0)},
               {1, std::complex<double>(1 / std::numbers::sqrt2, 0)}}},
             {1,
              {{0, std::complex<double>(1 / std::numbers::sqrt2, 0)},
               {1, std::complex<double>(-1 / std::numbers::sqrt2, 0)}}}}};
      })
      .Case<mlir::qco::XOp>([&](auto) {
        return ResultMap{{{0, {{1, std::complex<double>(1, 0)}}},
                          {1, {{0, std::complex<double>(1, 0)}}}}};
      })
      .Case<mlir::qco::YOp>([&](auto) {
        return ResultMap{{{0, {{1, std::complex<double>(0, 1)}}},
                          {1, {{0, std::complex<double>(0, -1)}}}}};
      })
      .Case<mlir::qco::ZOp>([&](auto) {
        return ResultMap{{{0, {{0, std::complex<double>(1, 0)}}},
                          {1, {{1, std::complex<double>(-1, 0)}}}}};
      })
      .Case<mlir::qco::SOp>([&](auto) {
        return ResultMap{{{0, {{0, std::complex<double>(1, 0)}}},
                          {1, {{1, std::complex<double>(0, 1)}}}}};
      })
      .Case<mlir::qco::SdgOp>([&](auto) {
        return ResultMap{{{0, {{0, std::complex<double>(1, 0)}}},
                          {1, {{1, std::complex<double>(0, -1)}}}}};
      })
      .Case<mlir::qco::TOp>([&](auto) {
        return ResultMap{
            {{0, {{0, std::complex<double>(1, 0)}}},
             {1,
              {{1, std::complex<double>(1 / std::numbers::sqrt2,
                                        1 / std::numbers::sqrt2)}}}}};
      })
      .Case<mlir::qco::TdgOp>([&](auto) {
        return ResultMap{
            {{0, {{0, std::complex<double>(1, 0)}}},
             {1,
              {{1, std::complex<double>(1 / std::numbers::sqrt2,
                                        -1 / std::numbers::sqrt2)}}}}};
      })
      .Case<mlir::qco::SXOp>([&](auto) {
        return ResultMap{{{0,
                           {{0, std::complex<double>(1.0 / 2.0, 1.0 / 2.0)},
                            {1, std::complex<double>(1.0 / 2.0, -1.0 / 2.0)}}},
                          {1,
                           {{0, std::complex<double>(1.0 / 2.0, -1.0 / 2.0)},
                            {1, std::complex<double>(1.0 / 2.0, 1.0 / 2.0)}}}}};
      })
      .Case<mlir::qco::SXdgOp>([&](auto) {
        return ResultMap{
            {{0,
              {{0, std::complex<double>(1.0 / 2.0, -1.0 / 2.0)},
               {1, std::complex<double>(1.0 / 2.0, 1.0 / 2.0)}}},
             {1,
              {{0, std::complex<double>(1.0 / 2.0, 1.0 / 2.0)},
               {1, std::complex<double>(1.0 / 2.0, -1.0 / 2.0)}}}}};
      })
      .Case<mlir::qco::RXOp>([&](auto) {
        return ResultMap{
            {{0,
              {{0, std::complex<double>(cos(params[0] / 2), 0)},
               {1, std::complex<double>(0, -sin(params[0] / 2))}}},
             {1,
              {{0, std::complex<double>(0, -sin(params[0] / 2))},
               {1, std::complex<double>(cos(params[0] / 2), 0)}}}}};
      })
      .Case<mlir::qco::RYOp>([&](auto) {
        return ResultMap{
            {{0,
              {{0, std::complex<double>(cos(params[0] / 2), 0)},
               {1, std::complex<double>(sin(params[0] / 2), 0)}}},
             {1,
              {{0, std::complex<double>(-sin(params[0] / 2), 0)},
               {1, std::complex<double>(cos(params[0] / 2), 0)}}}}};
      })
      .Case<mlir::qco::RZOp>([&](auto) {
        return ResultMap{
            {{0, {{0, exp(std::complex<double>(0, -params[0] / 2))}}},
             {1, {{1, exp(std::complex<double>(0, params[0] / 2))}}}}};
      })
      .Case<mlir::qco::POp>([&](auto) {
        return ResultMap{{{0, {{0, std::complex<double>(1, 0)}}},
                          {1, {{1, exp(std::complex<double>(0, params[0]))}}}}};
      })
      .Case<mlir::qco::ROp>([&](auto) {
        return ResultMap{
            {{0,
              {{0, std::complex<double>(cos(params[0] / 2), 0)},
               {1, exp(std::complex<double>(0, params[1])) *
                       std::complex<double>(0, -sin(params[0] / 2))}}},
             {1,
              {{0, exp(std::complex<double>(0, -params[1])) *
                       std::complex<double>(0, -sin(params[0] / 2))},
               {1, std::complex<double>(cos(params[0] / 2), 0)}}}}};
      })
      .Case<mlir::qco::U2Op>([&](auto) {
        return ResultMap{
            {{0,
              {{0, std::complex<double>(1 / std::numbers::sqrt2, 0)},
               {1, exp(std::complex<double>(0, params[0]))}}},
             {1,
              {{0, -exp(std::complex<double>(0, params[1]))},
               {1, exp(std::complex<double>(0, params[0] + params[1]))}}}}};
      })
      .Case<mlir::qco::UOp>([&](auto) {
        return ResultMap{
            {{0,
              {{0, std::complex<double>(cos(params[0] / 2), 0)},
               {1,
                exp(std::complex<double>(0, params[1])) * sin(params[0] / 2)}}},
             {1,
              {{0,
                -exp(std::complex<double>(0, params[2])) * sin(params[0] / 2)},
               {1, exp(std::complex<double>(0, params[1] + params[2])) *
                       cos(params[0] / 2)}}}}};
      })
      .Case<mlir::qco::SWAPOp>([&](auto) {
        return ResultMap{{{0, {{0, std::complex<double>(1, 0)}}},
                          {1, {{2, std::complex<double>(1, 0)}}},
                          {2, {{1, std::complex<double>(1, 0)}}},
                          {3, {{3, std::complex<double>(1, 0)}}}}};
      })
      .Case<mlir::qco::iSWAPOp>([&](auto) {
        return ResultMap{{{0, {{0, std::complex<double>(1, 0)}}},
                          {1, {{2, std::complex<double>(0, -1)}}},
                          {2, {{1, std::complex<double>(0, -1)}}},
                          {3, {{3, std::complex<double>(1, 0)}}}}};
      })
      .Case<mlir::qco::DCXOp>([&](auto) {
        return ResultMap{{{0, {{0, std::complex<double>(1, 0)}}},
                          {1, {{2, std::complex<double>(1, 0)}}},
                          {2, {{3, std::complex<double>(1, 0)}}},
                          {3, {{1, std::complex<double>(1, 0)}}}}};
      })
      .Case<mlir::qco::ECROp>([&](auto) {
        return ResultMap{
            {{0,
              {{2, std::complex<double>(1 / std::numbers::sqrt2, 0)},
               {3, std::complex<double>(0, -1 / std::numbers::sqrt2)}}},
             {1,
              {{2, std::complex<double>(0, -1 / std::numbers::sqrt2)},
               {3, std::complex<double>(1 / std::numbers::sqrt2, 0)}}},
             {2,
              {{0, std::complex<double>(1 / std::numbers::sqrt2, 0)},
               {1, std::complex<double>(0, 1 / std::numbers::sqrt2)}}},
             {3,
              {{0, std::complex<double>(0, 1 / std::numbers::sqrt2)},
               {1, std::complex<double>(1 / std::numbers::sqrt2, 0)}}}}};
      })
      .Case<mlir::qco::RXXOp>([&](auto) {
        return ResultMap{
            {{0,
              {{0, std::complex<double>(cos(params[0] / 2), 0)},
               {3, std::complex<double>(0, -sin(params[0] / 2))}}},
             {1,
              {{1, std::complex<double>(cos(params[0] / 2), 0)},
               {2, std::complex<double>(0, -sin(params[0] / 2))}}},
             {2,
              {{1, std::complex<double>(0, -sin(params[0] / 2))},
               {2, std::complex<double>(cos(params[0] / 2), 0)}}},
             {3,
              {{0, std::complex<double>(0, -sin(params[0] / 2))},
               {3, std::complex<double>(cos(params[0] / 2), 0)}}}}};
      })
      .Case<mlir::qco::RYYOp>([&](auto) {
        return ResultMap{
            {{0,
              {{0, std::complex<double>(cos(params[0] / 2), 0)},
               {3, std::complex<double>(0, sin(params[0] / 2))}}},
             {1,
              {{1, std::complex<double>(cos(params[0] / 2), 0)},
               {2, std::complex<double>(0, -sin(params[0] / 2))}}},
             {2,
              {{1, std::complex<double>(0, -sin(params[0] / 2))},
               {2, std::complex<double>(cos(params[0] / 2), 0)}}},
             {3,
              {{0, std::complex<double>(0, sin(params[0] / 2))},
               {3, std::complex<double>(cos(params[0] / 2), 0)}}}}};
      })
      .Case<mlir::qco::RZXOp>([&](auto) {
        return ResultMap{
            {{0,
              {{0, std::complex<double>(cos(params[0] / 2), 0)},
               {1, std::complex<double>(0, -sin(params[0] / 2))}}},
             {1,
              {{0, std::complex<double>(0, -sin(params[0] / 2))},
               {1, std::complex<double>(cos(params[0] / 2), 0)}}},
             {2,
              {{2, std::complex<double>(cos(params[0] / 2), 0)},
               {3, std::complex<double>(0, sin(params[0] / 2))}}},
             {3,
              {{2, std::complex<double>(0, sin(params[0] / 2))},
               {3, std::complex<double>(cos(params[0] / 2), 0)}}}}};
      })
      .Case<mlir::qco::RZZOp>([&](auto) {
        return ResultMap{
            {{0, {{0, exp(std::complex<double>(0, -params[0] / 2))}}},
             {1, {{1, exp(std::complex<double>(0, params[0] / 2))}}},
             {2, {{2, exp(std::complex<double>(0, params[0] / 2))}}},
             {3, {{3, exp(std::complex<double>(0, -params[0] / 2))}}}}};
      })
      .Case<mlir::qco::XXPlusYYOp>([&](auto) {
        return ResultMap{{{0, {{0, std::complex<double>(1, 0)}}},
                          {1,
                           {{1, std::complex<double>(cos(params[0] / 2), 0)},
                            {2, std::complex<double>(0, -sin(params[0] / 2)) *
                                    exp(std::complex<double>(0, params[1]))}}},
                          {2,
                           {{1, std::complex<double>(0, -sin(params[0] / 2)) *
                                    exp(std::complex<double>(0, -params[1]))},
                            {2, std::complex<double>(cos(params[0] / 2), 0)}}},
                          {3, {{3, std::complex<double>(1, 0)}}}}};
      })
      .Case<mlir::qco::XXMinusYYOp>([&](auto) {
        return ResultMap{
            {{0,
              {{0, std::complex<double>(cos(params[0] / 2), 0)},
               {3, std::complex<double>(0, -sin(params[0] / 2)) *
                       exp(std::complex<double>(0, params[1]))}}},
             {1, {{1, std::complex<double>(1, 0)}}},
             {2, {{2, std::complex<double>(1, 0)}}},
             {3,
              {{0, std::complex<double>(0, -sin(params[0] / 2)) *
                       exp(std::complex<double>(0, -params[1]))},
               {3, std::complex<double>(cos(params[0] / 2), 0)}}}}};
      })
      .Default([&](auto) -> ResultMap {
        throw std::runtime_error("Unsupported gate in mlir::qco::gatetomap");
      });
}

#endif // MQT_CORE_GATETOMAP_H