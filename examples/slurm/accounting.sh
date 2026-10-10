#!/bin/sh
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set -eu

cluster=$(sacctmgr --noheader --parsable2 show cluster mqt-core format=Cluster)
if [ -z "$cluster" ]; then
    sacctmgr --immediate add cluster mqt-core
fi
account=$(sacctmgr --noheader --parsable2 show account research format=Account)
if [ -z "$account" ]; then
    sacctmgr --immediate add account research Description="QDMI workloads" Organization=MQT
fi
association=$(sacctmgr --noheader --parsable2 show association where Cluster=mqt-core Account=research User=mqt-test format=User)
if [ -z "$association" ]; then
    sacctmgr --immediate add user mqt-test Cluster=mqt-core Account=research DefaultAccount=research
fi
