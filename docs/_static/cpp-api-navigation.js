// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

document.addEventListener("DOMContentLoaded", () => {
  const backlink = document.getElementById("mqt-docs-backlink");
  const returnUrl = sessionStorage.getItem("mqt-core-cpp-return-url");
  if (backlink && returnUrl) {
    backlink.href = returnUrl;
  }
});
