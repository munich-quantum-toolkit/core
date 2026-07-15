// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

document.addEventListener("DOMContentLoaded", () => {
    const backlink = document.getElementById("mqt-docs-backlink");
    if (backlink) {
        const returnUrl = sessionStorage.getItem("mqt-core-cpp-return-url");
        if (returnUrl) {
            backlink.href = returnUrl;
        }
        return;
    }

    document.addEventListener("click", (event) => {
        const link = event.target.closest("a[href]");
        if (!link) return;
        const target = new URL(link.href, window.location.href);
        if (
            target.origin === window.location.origin &&
            target.pathname.includes("/cpp/")
        ) {
            sessionStorage.setItem("mqt-core-cpp-return-url", window.location.href);
        }
    });
});
