/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#ifndef MQT_CORE_GATETOMAP_H
#define MQT_CORE_GATETOMAP_H

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Drivers.h"

#include <cmath>
#include <complex>
#include <numbers>
#include <span>
#include <stdexcept>
#include <unordered_map>

/**
 * This file provides information of available quantum gates as mappings. It is
 * used in constant propagation to get the factors each amplitude has to be
 * multiplied with to get the new amplitudes after a gate application.
 */

using Complex = std::complex<double>;

using ResultMap =
    std::unordered_map<unsigned int, std::unordered_map<unsigned int, Complex>>;

constexpr double inv_sqrt2 = 1.0 / std::numbers::sqrt2;

inline std::unordered_map<unsigned int,
                          std::unordered_map<unsigned int, Complex>>
getQubitMappingOfGates(mlir::Operation* gate, const std::span<double>& params) {

  return mlir::TypeSwitch<
             mlir::Operation*,
             std::unordered_map<unsigned int,
                                std::unordered_map<unsigned int, Complex>>>(
             gate)
      .Case<mlir::qco::IdOp>([&](auto) {
        return ResultMap{{0, {{0, Complex(1, 0)}}}, {1, {{1, Complex(1, 0)}}}};
      })
      .Case<mlir::qco::HOp>([&](auto) {
        return ResultMap{
            {{0, {{0, Complex(inv_sqrt2, 0)}, {1, Complex(inv_sqrt2, 0)}}},
             {1, {{0, Complex(inv_sqrt2, 0)}, {1, Complex(-inv_sqrt2, 0)}}}}};
      })
      .Case<mlir::qco::XOp>([&](auto) {
        return ResultMap{
            {{0, {{1, Complex(1, 0)}}}, {1, {{0, Complex(1, 0)}}}}};
      })
      .Case<mlir::qco::YOp>([&](auto) {
        return ResultMap{
            {{0, {{1, Complex(0, 1)}}}, {1, {{0, Complex(0, -1)}}}}};
      })
      .Case<mlir::qco::ZOp>([&](auto) {
        return ResultMap{
            {{0, {{0, Complex(1, 0)}}}, {1, {{1, Complex(-1, 0)}}}}};
      })
      .Case<mlir::qco::SOp>([&](auto) {
        return ResultMap{
            {{0, {{0, Complex(1, 0)}}}, {1, {{1, Complex(0, 1)}}}}};
      })
      .Case<mlir::qco::SdgOp>([&](auto) {
        return ResultMap{
            {{0, {{0, Complex(1, 0)}}}, {1, {{1, Complex(0, -1)}}}}};
      })
      .Case<mlir::qco::TOp>([&](auto) {
        return ResultMap{{{0, {{0, Complex(1, 0)}}},
                          {1, {{1, Complex(inv_sqrt2, inv_sqrt2)}}}}};
      })
      .Case<mlir::qco::TdgOp>([&](auto) {
        return ResultMap{{{0, {{0, Complex(1, 0)}}},
                          {1, {{1, Complex(inv_sqrt2, -inv_sqrt2)}}}}};
      })
      .Case<mlir::qco::SXOp>([&](auto) {
        return ResultMap{
            {{0, {{0, Complex(0.5, 0.5)}, {1, Complex(0.5, -0.5)}}},
             {1, {{0, Complex(0.5, -0.5)}, {1, Complex(0.5, 0.5)}}}}};
      })
      .Case<mlir::qco::SXdgOp>([&](auto) {
        return ResultMap{
            {{0, {{0, Complex(0.5, -0.5)}, {1, Complex(0.5, 0.5)}}},
             {1, {{0, Complex(0.5, 0.5)}, {1, Complex(0.5, -0.5)}}}}};
      })
      .Case<mlir::qco::RXOp>([&](auto) {
        const double c = cos(0.5 * params[0]);
        const double s = sin(0.5 * params[0]);
        return ResultMap{{{0, {{0, Complex(c, 0)}, {1, Complex(0, -s)}}},
                          {1, {{0, Complex(0, -s)}, {1, Complex(c, 0)}}}}};
      })
      .Case<mlir::qco::RYOp>([&](auto) {
        const double c = cos(0.5 * params[0]);
        const double s = sin(0.5 * params[0]);
        return ResultMap{{{0, {{0, Complex(c, 0)}, {1, Complex(s, 0)}}},
                          {1, {{0, Complex(-s, 0)}, {1, Complex(c, 0)}}}}};
      })
      .Case<mlir::qco::RZOp>([&](auto) {
        const double halfParameter = 0.5 * params[0];
        return ResultMap{{{0, {{0, exp(Complex(0, -halfParameter))}}},
                          {1, {{1, exp(Complex(0, halfParameter))}}}}};
      })
      .Case<mlir::qco::POp>([&](auto) {
        return ResultMap{{{0, {{0, Complex(1, 0)}}},
                          {1, {{1, exp(Complex(0, params[0]))}}}}};
      })
      .Case<mlir::qco::ROp>([&](auto) {
        const double c = cos(0.5 * params[0]);
        const double s = sin(0.5 * params[0]);
        return ResultMap{{{0,
                           {{0, Complex(c, 0)},
                            {1, exp(Complex(0, params[1])) * Complex(0, -s)}}},
                          {1,
                           {{0, exp(Complex(0, -params[1])) * Complex(0, -s)},
                            {1, Complex(c, 0)}}}}};
      })
      .Case<mlir::qco::U2Op>([&](auto) {
        return ResultMap{
            {{0, {{0, Complex(inv_sqrt2, 0)}, {1, exp(Complex(0, params[0]))}}},
             {1,
              {{0, -exp(Complex(0, params[1]))},
               {1, exp(Complex(0, params[0] + params[1]))}}}}};
      })
      .Case<mlir::qco::UOp>([&](auto) {
        const double c = cos(0.5 * params[0]);
        const double s = sin(0.5 * params[0]);
        return ResultMap{
            {{0, {{0, Complex(c, 0)}, {1, exp(Complex(0, params[1])) * s}}},
             {1,
              {{0, -exp(Complex(0, params[2])) * s},
               {1, exp(Complex(0, params[1] + params[2])) * c}}}}};
      })
      .Case<mlir::qco::SWAPOp>([&](auto) {
        return ResultMap{{{0, {{0, Complex(1, 0)}}},
                          {1, {{2, Complex(1, 0)}}},
                          {2, {{1, Complex(1, 0)}}},
                          {3, {{3, Complex(1, 0)}}}}};
      })
      .Case<mlir::qco::iSWAPOp>([&](auto) {
        return ResultMap{{{0, {{0, Complex(1, 0)}}},
                          {1, {{2, Complex(0, -1)}}},
                          {2, {{1, Complex(0, -1)}}},
                          {3, {{3, Complex(1, 0)}}}}};
      })
      .Case<mlir::qco::DCXOp>([&](auto) {
        return ResultMap{{{0, {{0, Complex(1, 0)}}},
                          {1, {{2, Complex(1, 0)}}},
                          {2, {{3, Complex(1, 0)}}},
                          {3, {{1, Complex(1, 0)}}}}};
      })
      .Case<mlir::qco::ECROp>([&](auto) {
        return ResultMap{
            {{0, {{2, Complex(inv_sqrt2, 0)}, {3, Complex(0, -inv_sqrt2)}}},
             {1, {{2, Complex(0, -inv_sqrt2)}, {3, Complex(inv_sqrt2, 0)}}},
             {2, {{0, Complex(inv_sqrt2, 0)}, {1, Complex(0, inv_sqrt2)}}},
             {3, {{0, Complex(0, inv_sqrt2)}, {1, Complex(inv_sqrt2, 0)}}}}};
      })
      .Case<mlir::qco::RXXOp>([&](auto) {
        const double c = cos(0.5 * params[0]);
        const double s = sin(0.5 * params[0]);
        return ResultMap{{{0, {{0, Complex(c, 0)}, {3, Complex(0, -s)}}},
                          {1, {{1, Complex(c, 0)}, {2, Complex(0, -s)}}},
                          {2, {{1, Complex(0, -s)}, {2, Complex(c, 0)}}},
                          {3, {{0, Complex(0, -s)}, {3, Complex(c, 0)}}}}};
      })
      .Case<mlir::qco::RYYOp>([&](auto) {
        const double c = cos(0.5 * params[0]);
        const double s = sin(0.5 * params[0]);
        return ResultMap{{{0, {{0, Complex(c, 0)}, {3, Complex(0, s)}}},
                          {1, {{1, Complex(c, 0)}, {2, Complex(0, -s)}}},
                          {2, {{1, Complex(0, -s)}, {2, Complex(c, 0)}}},
                          {3, {{0, Complex(0, s)}, {3, Complex(c, 0)}}}}};
      })
      .Case<mlir::qco::RZXOp>([&](auto) {
        const double c = cos(0.5 * params[0]);
        const double s = sin(0.5 * params[0]);
        return ResultMap{{{0, {{0, Complex(c, 0)}, {1, Complex(0, -s)}}},
                          {1, {{0, Complex(0, -s)}, {1, Complex(c, 0)}}},
                          {2, {{2, Complex(c, 0)}, {3, Complex(0, s)}}},
                          {3, {{2, Complex(0, s)}, {3, Complex(c, 0)}}}}};
      })
      .Case<mlir::qco::RZZOp>([&](auto) {
        const double halfParam = 0.5 * params[0];
        Complex ePos = exp(Complex(0, halfParam));
        Complex eNeg = exp(Complex(0, -halfParam));
        return ResultMap{{{0, {{0, eNeg}}},
                          {1, {{1, ePos}}},
                          {2, {{2, ePos}}},
                          {3, {{3, eNeg}}}}};
      })
      .Case<mlir::qco::XXPlusYYOp>([&](auto) {
        const double c = cos(0.5 * params[0]);
        const double s = sin(0.5 * params[0]);
        return ResultMap{{{0, {{0, Complex(1, 0)}}},
                          {1,
                           {{1, Complex(c, 0)},
                            {2, Complex(0, -s) * exp(Complex(0, params[1]))}}},
                          {2,
                           {{1, Complex(0, -s) * exp(Complex(0, -params[1]))},
                            {2, Complex(c, 0)}}},
                          {3, {{3, Complex(1, 0)}}}}};
      })
      .Case<mlir::qco::XXMinusYYOp>([&](auto) {
        const double c = cos(0.5 * params[0]);
        const double s = sin(0.5 * params[0]);
        return ResultMap{{{0,
                           {{0, Complex(c, 0)},
                            {3, Complex(0, -s) * exp(Complex(0, params[1]))}}},
                          {1, {{1, Complex(1, 0)}}},
                          {2, {{2, Complex(1, 0)}}},
                          {3,
                           {{0, Complex(0, -s) * exp(Complex(0, -params[1]))},
                            {3, Complex(c, 0)}}}}};
      })
      .Default([&](auto) -> ResultMap {
        throw std::runtime_error("Unsupported gate in mlir::qco::gatetomap");
      });
}

#endif // MQT_CORE_GATETOMAP_H
