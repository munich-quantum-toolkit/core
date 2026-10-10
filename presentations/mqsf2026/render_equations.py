# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# /// script
# dependencies = ["matplotlib==3.11.2", "segno==1.6.6"]
# ///
"""Render the keynote equations as reproducible SVG paths and update provenance."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import segno

ASSETS = Path(__file__).resolve().parent / "assets"
EQUATIONS = {
    "eq-afqmc-projection": (
        "Imaginary-time propagation of the trial state",
        [r"$|\Psi(\tau)\rangle = e^{-\tau(\hat{H}-E_T)}\,|\Psi_T\rangle$"],
        "Formal unnormalized imaginary-time propagation. The phaseless AFQMC calculation approximates this projection.",
    ),
    "eq-afqmc-energy": (
        "Trial-state local energy and its weighted walker estimate",
        [
            (
                r"$E_L[\phi_w]=\frac{\langle\Psi_T|\hat{H}|\phi_w\rangle}"
                r"{\langle\Psi_T|\phi_w\rangle}$"
            ),
            (
                r"$\widehat{E}(\tau)=\frac{\sum_w W_w(\tau)\,\mathrm{Re}\,E_L[\phi_w(\tau)]}"
                r"{\sum_w W_w(\tau)}$"
            ),
        ],
        (
            "The local-energy ratio requires nonzero trial overlap. "
            "The displayed estimator uses nonnegative walker weights."
        ),
    ),
    "eq-qpe-phase": (
        "Quantum phase estimation: eigenphase and measured binary fraction",
        [
            r"$\hat{U}|\psi\rangle=e^{2\pi i\varphi}|\psi\rangle$",
            (
                r"$\varphi\approx 0.b_1b_2\ldots b_m"
                r"=\sum_{k=1}^{m}\frac{b_k}{2^k}=\frac{j}{2^m}$"
            ),
        ],
        (
            "The binary fraction is exact only for a representable phase; "
            "finite-register QPE samples nearby integer outcomes."
        ),
    ),
}


def main() -> None:
    """Write path-only equation assets and the Core repository QR code."""
    mpl.rcParams.update({
        "svg.fonttype": "path",
        "svg.hashsalt": "mqsf2026-equations",
        "mathtext.fontset": "stix",
    })
    sources = json.loads((ASSETS / "sources.json").read_text())
    for name, (label, equations, scope) in EQUATIONS.items():
        fig = plt.figure(figsize=(16, 1.25 * len(equations)), facecolor="white")
        for row, formula in enumerate(equations):
            fig.text(
                0.5,
                1 - (row + 0.5) / len(equations),
                formula,
                color="#142B45" if row == 0 else "#2F70B8",
                fontsize=40,
                horizontalalignment="center",
                verticalalignment="center",
            )
        output = io.StringIO()
        fig.savefig(
            output,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.04,
            metadata={"Date": None, "Creator": f"Matplotlib {mpl.__version__}"},
        )
        plt.close(fig)
        svg = output.getvalue()
        svg = re.sub(r'\s(?:width|height)="[^"]*"', "", svg, count=2)
        svg = svg.replace("<svg ", f'<svg role="img" aria-label="{label}" ', 1)
        svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
        assert "<text" not in svg
        target = ASSETS / f"{name}.svg"
        target.write_text(svg)
        sources[name] = {
            "generator": (
                f"render_equations.py; Matplotlib {mpl.__version__}; STIX mathtext; "
                "SVG path glyphs; tight text bounds with 0.04-inch padding"
            ),
            "equations": equations,
            "scope": scope,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    target = ASSETS / "qr-core-repo.svg"
    url = "https://github.com/munich-quantum-toolkit/core"
    segno.make(url, error="m", micro=False).save(
        target, border=4, scale=1, dark="#142b45", light="white", xmldecl=False
    )
    sources["qr-core-repo"] = {
        "encoded_url": url,
        "generator": "render_equations.py; Segno 1.6.6; micro=False, error=m, border=4, navy on white",
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (ASSETS / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
