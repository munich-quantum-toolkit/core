/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

"use strict";
(() => {
  const data = window.MQSF_DATA;
  const assets = window.MQSF_ASSETS;
  const viz = window.MQSF_VIZ;
  const $ = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  const scenario = (id) => data.scenarios.find((s) => s.id === id);
  const variant = (id = "parity", unrolled = false) =>
    scenario(id)?.variants.find(
      (v) => v.id === (unrolled ? "unrolled" : "structured"),
    ) || {};
  const app = data.application || {};
  const targets = data.targets || [];
  const references = data.references || {};
  const run =
    variant("qpe").executions?.["qir-adaptive"] ||
    variant("qpe").execution ||
    {};
  const state = {
    slide: 0,
    step: 0,
    animation: null,
    token: 0,
    jump: "",
    overview: 0,
  };
  const svg = (body, box = "0 0 1600 650", name = "Illustration") =>
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${box}" role="img" aria-label="${esc(name)}">${body}</svg>`;
  const text = (x, y, s, size = 32, color = "#142b45", anchor = "middle") =>
    `<text x="${x}" y="${y}" font-family="Inter,Arial,sans-serif" font-size="${size}" fill="${color}" text-anchor="${anchor}">${esc(s)}</text>`;
  const line = (x1, y1, x2, y2, color = "#2f70b8", width = 4) =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"/>`;
  const logo = (name, cls = "logo") =>
    `<img class="${cls}" src="${assets[name]}" alt="${esc(name === "tum-cda" ? "Chair for Design Automation, TUM" : name.toUpperCase())}" />`;
  const heading = (title, sub = "") =>
    `<h2 data-morph="heading">${title}</h2>${sub ? `<p class="subtitle">${sub}</p>` : ""}`;
  const qr = (name, url, caption, showStats = true) => {
    const stats = showStats && references.repo_stats?.[`https://${url}`];
    const stars = stats
      ? `<span class="repo-stars"><svg viewBox="0 0 24 24" aria-label="GitHub stars"><path d="m12 2 3.1 6.3 6.9 1-5 4.9 1.2 6.9-6.2-3.3-6.2 3.3L7 14.2 2 9.3l6.9-1Z" fill="#e7ae18"/></svg>${stats.stars}${stats.downloads ? ` · ${esc(stats.downloads)}` : ""}</span>`
      : "";
    return `<div class="qr-block" data-morph="qr-${name}"><img src="${assets[`qr-${name}`]}" alt="QR code: ${esc(url)}"/><div><div class="url">${url.startsWith("github.com/") ? logo("github", "github-mark") + esc(url.slice(11)) : esc(url)}</div><div class="caption">${esc(caption)}</div>${stars}</div></div>`;
  };
  const fmt = (n, digits = 2) =>
    Number(n).toLocaleString("en-US", { maximumFractionDigits: digits });
  function code(
    a,
    { start = 0, count = 11, hot = [], title, compact = false, key } = {},
  ) {
    if (!a?.code)
      return `<div class="code-frame"><p class="small">Capture unavailable</p></div>`;
    const lines = a.code.split("\n");
    start = Math.max(0, Math.min(start, Math.max(0, lines.length - 1)));
    const shown = lines.slice(start, start + count);
    const indent = Math.min(
      ...shown.filter((l) => l.trim()).map((l) => l.match(/^ */)[0].length),
      100,
    );
    const body = shown
      .map((raw, i) => {
        const l = raw.slice(indent),
          highlighted = a.lines_html?.[start + i]?.replace(
            new RegExp(`^ {0,${indent}}`),
            "",
          );
        const content = highlighted ?? esc(l);
        return `<span class="code-line ${hot.includes(start + i) ? "hot" : ""}"><span class="line-no">${start + i + (a.excerpt_start_line || 1)}</span><span class="code-text">${content}</span></span>`;
      })
      .join("");
    return `<div data-morph="code-${esc(key || a.id || title || a.label || a.language || "source")}" class="code-frame ${compact ? "compact" : ""}"><div class="code-title">${esc(title || a.label || a.language || "Actual source")}</div><pre class="highlight">${body}</pre></div>`;
  }
  const focus = (a, needle, before = 1) =>
    Math.max(
      0,
      (a?.code?.split("\n").findIndex((s) => s.includes(needle)) ?? 0) - before,
    );
  const flatten = window.MQSF_CIRCUIT.flatten;
  const circuit = window.MQSF_CIRCUIT.render;
  function histogram(capture, elapsed = Infinity, box = "0 0 900 380") {
    const bins = new Map();
    const timed = capture.shot_events || [];
    const shots = Number.isFinite(elapsed)
      ? timed.filter((s) => s.time_ms <= elapsed).map((s) => s.outcome)
      : capture.shots || [];
    shots.forEach((s) =>
      bins.set(parseInt(s, 2), (bins.get(parseInt(s, 2)) || 0) + 1),
    );
    const totals = capture.counts || {},
      max = Math.max(1, ...Object.values(totals));
    const all = Object.keys(totals).map((s) => parseInt(s, 2));
    // Keep the visible QPE window stable; aggregate real tails instead of hiding them.
    const lo = 78,
      hi = 93,
      values = [
        bins.get(-1) || 0,
        ...Array.from({ length: hi - lo + 1 }, (_, i) => bins.get(lo + i) || 0),
        0,
      ];
    values[0] = [...bins]
      .filter(([i]) => i < lo)
      .reduce((s, [, n]) => s + n, 0);
    values[values.length - 1] = [...bins]
      .filter(([i]) => i > hi)
      .reduce((s, [, n]) => s + n, 0);
    const x0 = 90,
      w = 765,
      plotH = 210,
      y0 = 250,
      bw = w / values.length;
    let body = line(x0, y0, x0 + w, y0, "#9cb3c9", 2);
    [0, 0.5, 1].forEach((f) => {
      const y = y0 - f * plotH;
      body +=
        line(x0, y, x0 + w, y, "#e0e9f2", 1) +
        text(x0 - 12, y + 7, String(Math.round(max * f)), 25, "#5b7088", "end");
    });
    const latest = shots.length ? parseInt(shots.at(-1), 2) : -100;
    values.forEach((n, i) => {
      const x = x0 + i * bw + 5,
        h = (n / max) * plotH,
        key =
          i === 0
            ? `<${lo}`
            : i === values.length - 1
              ? `>${hi}`
              : String(lo + i - 1);
      body += `<rect class="bar ${Number(key) === latest ? "hot" : ""}" x="${x}" y="${y0 - h}" width="${bw - 9}" height="${h}" data-count="${n}"/>`;
      if (n > max * 0.07) body += text(x + (bw - 9) / 2, y0 - h - 12, n, 25);
      if (
        i === 0 ||
        i === values.length - 1 ||
        (i % 2 === 0 && i < values.length - 2) ||
        key === "85"
      )
        body += text(x + (bw - 9) / 2, y0 + 32, key, 25, "#5b7088");
    });
    body += text(450, 318, "Phase Estimate · Integer / 256", 26, "#5b7088");
    return `<div class="histogram" data-shots="${shots.length}" data-elapsed-ms="${elapsed}" data-observed-bins="${all.length}">${svg(body, box, "Actual QPE measurement distribution")}</div>`;
  }
  const appStage = (id) => {
    const v = targets[0]?.compilation || {};
    return [...(v.stages || []), ...(v.exports || [])].find((a) => a.id === id);
  };
  const stagesFor = (target) => [
    ...(target?.compilation?.stages || []),
    ...(target?.compilation?.exports || []),
  ];
  const targetStage = (target, id) =>
    stagesFor(target).find((a) => a.id === id);
  function recordedPosition(times, elapsed) {
    if (!times.length || elapsed <= times[0]) return 0;
    const next = times.findIndex((time) => time > elapsed);
    if (next < 0) return times.length - 1;
    return (
      next - 1 + (elapsed - times[next - 1]) / (times[next] - times[next - 1])
    );
  }
  function walkerScene(progress = 1, elapsed = null, trial = 0) {
    const frames = app.propagation?.curves?.[trial]?.frames || [];
    if (!frames.length) return "";
    const index =
        elapsed === null
          ? Math.max(
              0,
              Math.min(frames.length - 1, progress * (frames.length - 1)),
            )
          : recordedPosition(
              frames.map((frame) =>
                Math.max(...frame.walkers.map((w) => w.completed_ms)),
              ),
              elapsed,
            ),
      low = Math.floor(index),
      high = Math.min(low + 1, frames.length - 1),
      fraction = index - low,
      current = frames[low],
      next = frames[high],
      tau = current.tau + (next.tau - current.tau) * fraction,
      weights = current.walkers.map(
        (walker, i) =>
          walker.weight + (next.walkers[i].weight - walker.weight) * fraction,
      ),
      max = Math.max(...weights),
      color = trial ? "#778899" : "#2f70b8";
    // Rings retain every walker identity, even when its measured weight approaches zero.
    let body = current.walkers
      .map((walker, i) => {
        const x = 48 + (i % 32) * 34,
          y = 30 + Math.floor(i / 32) * 34,
          radius = 14 * Math.sqrt(weights[i] / max);
        return `<g data-morph="walker-${trial}-${walker.id}" data-trial="${trial}" data-walker-id="${walker.id}" data-weight="${weights[i]}"><circle cx="${x}" cy="${y}" r="14" fill="none" stroke="#cbd9e6" stroke-width="1.5"/><circle cx="${x}" cy="${y}" r="${radius}" fill="${color}"/></g>`;
      })
      .join("");
    body += text(
      580,
      195,
      `τ = ${fmt(tau, 2)} Ha⁻¹ · filled area ∝ weight`,
      30,
      color,
    );
    return svg(
      body,
      "0 0 1160 220",
      `${trial ? "Hartree–Fock" : "Quantum-assisted"} trial: interpolated recorded walker states`,
    );
  }

  function deviceView(
    target = targets[0],
    activeSites = [],
    layout = null,
    blend = null,
    activeEdges = null,
  ) {
    const meta = target?.metadata || data.device,
      sites = meta.sites || [],
      edges = meta.edges || [];
    const full = edges.length > sites.length * 5,
      points = new Map();
    if (full)
      sites.forEach((s, i) =>
        points.set(s.id, {
          x: 410 + 245 * Math.cos((i * 2 * Math.PI) / sites.length),
          y: 310 + 245 * Math.sin((i * 2 * Math.PI) / sites.length),
        }),
      );
    else {
      // The full Emerald lattice fixes drawing coordinates; only advertised edges are drawn.
      const geometry = target?.id === "iqm.emerald" ? data.device.edges : edges;
      const graph = new Set(
          geometry.flatMap(([a, b]) => [`${a}:${b}`, `${b}:${a}`]),
        ),
        rows = [];
      [...sites]
        .sort((a, b) => a.id - b.id)
        .forEach((s) => {
          const r = rows.at(-1);
          if (r && graph.has(`${r.at(-1).id}:${s.id}`)) r.push(s);
          else rows.push([s]);
        });
      rows.forEach((r, y) => {
        let offset = 0;
        const connection = r.flatMap((s, x) =>
          (rows[y - 1] || [])
            .filter((a) => graph.has(`${a.id}:${s.id}`))
            .map((a) => points.get(a.id).x - x),
        )[0];
        if (connection !== undefined) offset = connection;
        r.forEach((s, x) => points.set(s.id, { x: x + offset, y }));
      });
      const xs = [...points.values()].map((p) => p.x),
        minX = Math.min(...xs),
        maxX = Math.max(...xs),
        maxY = Math.max(1, rows.length - 1),
        scale = Math.min(640 / Math.max(1, maxX - minX), 470 / maxY);
      points.forEach((p) => {
        p.x = 410 + (p.x - (minX + maxX) / 2) * scale;
        p.y = 300 + (p.y - maxY / 2) * scale;
      });
    }
    const active = new Set(activeSites),
      colors = [
        "#2f70b8",
        "#087d92",
        "#8864bd",
        "#dd8d23",
        "#d46172",
        "#5b83a1",
      ];
    const fidelities = new Map(
      (target?.compiler_model?.operations || [])
        .filter((o) => o.numQubits === 2)
        .flatMap((o) =>
          (o.siteOverrides || [])
            .filter((v) => v.sites?.length === 2 && Number.isFinite(v.fidelity))
            .map((v) => [
              v.sites
                .slice()
                .sort((a, b) => a - b)
                .join(":"),
              v.fidelity,
            ]),
        ),
    );
    const fidelityColor = (f) =>
      f === undefined
        ? "#b8cee2"
        : f < 0.98
          ? "#ce9152"
          : f < 0.995
            ? "#54a0b2"
            : "#2f70b8";
    let body = edges
      .map(([a, b]) => {
        const p = points.get(a),
          q = points.get(b);
        if (!p || !q) return "";
        const f = fidelities.get([a, b].sort((x, y) => x - y).join(":")),
          hot =
            activeEdges === null
              ? active.has(a) && active.has(b)
              : activeEdges.has([a, b].sort((x, y) => x - y).join(":"));
        return `<line data-morph="device-edge-${a}-${b}" x1="${p.x}" y1="${p.y}" x2="${q.x}" y2="${q.y}" stroke="${hot ? "#087d92" : fidelityColor(f)}" stroke-width="${hot ? 6 : full ? 1 : 3.5}" opacity="${full ? 0.22 : 1}"><title>${a}–${b}${f === undefined ? "" : `: reported fidelity ${fmt(f * 100, 3)}%`}</title></line>`;
      })
      .join("");
    body += text(
      410,
      620,
      full ? "All-to-all connectivity" : "",
      24,
      "#526d88",
    );
    const r = sites.length > 80 ? 12 : sites.length > 40 ? 17 : 20;
    const tokens = (layout || [])
      .slice(0, 6)
      .map((site, i) => {
        const destination = points.get(site),
          origin = points.get(blend?.before?.[i]) || destination;
        if (!destination) return null;
        const t = blend?.fraction ?? 1,
          dx = destination.x - origin.x,
          dy = destination.y - origin.y,
          distance = Math.hypot(dx, dy),
          // Opposing exchanges take opposite arcs; recorded endpoints stay exact.
          arc = distance
            ? (Math.min(45, distance / 4) * Math.sin(Math.PI * t)) / distance
            : 0;
        return {
          x: origin.x + dx * t - dy * arc,
          y: origin.y + dy * t + dx * arc,
          i,
        };
      })
      .filter(Boolean);
    // Keep labels apart during schematic movement, never change recorded placements.
    if (blend?.fraction > 0 && blend.fraction < 1)
      for (let pass = 0; pass < 12; pass++)
        for (let i = 0; i < tokens.length; i++)
          for (let j = i + 1; j < tokens.length; j++) {
            const a = tokens[i],
              b = tokens[j],
              dx = b.x - a.x || 0.001,
              dy = b.y - a.y,
              distance = Math.hypot(dx, dy),
              shift = Math.max(
                0,
                (2 * (r + 5) + 4 - distance) / (2 * distance),
              );
            a.x -= dx * shift;
            a.y -= dy * shift;
            b.x += dx * shift;
            b.y += dy * shift;
          }
    body += sites
      .map((s) => {
        const p = points.get(s.id),
          hot = active.has(s.id);
        const covered = tokens.some(
          (t) => Math.hypot(t.x - p.x, t.y - p.y) < r + 18,
        );
        return `<g data-morph="device-node-${s.id}" transform="translate(${p.x} ${p.y})"><circle r="${r}" fill="${hot ? "#d4e9fa" : "#ecf3fa"}" stroke="${hot ? "#087d92" : "#92b2d0"}" stroke-width="${hot ? 4 : 1.5}"/>${covered ? "" : text(0, 6, s.id, r > 15 ? 17 : 12, "#58718b")}</g>`;
      })
      .join("");
    tokens.forEach(({ x, y, i }) => {
      body += `<g data-morph="logical-qubit-${i}" transform="translate(${x} ${y})"><circle r="${r + 5}" fill="${colors[i]}" stroke="white" stroke-width="3"/>${text(0, 7, `q${i}`, 17, "white")}</g>`;
    });
    const xs = [...points.values()].map((p) => p.x),
      ys = [...points.values()].map((p) => p.y),
      left = Math.min(...xs) - r - 10,
      top = Math.min(...ys) - r - 10,
      right = Math.max(...xs) + r + 10,
      bottom = full ? 632 : Math.max(...ys) + r + 10;
    return `<div class="target-map" data-morph="target-map"><svg xmlns="http://www.w3.org/2000/svg" data-morph="target-map-svg" viewBox="${left} ${top} ${right - left} ${bottom - top}" role="img" aria-label="${esc(target?.label || "Emerald")} captured coupling graph">${body}</svg></div>`;
  }
  function programMetrics(target, id) {
    const m = target?.compilation?.metrics?.[id] || {},
      c = targetStage(target, id)?.circuit,
      ops = c ? flatten(c) : [];
    const two =
      m.two_qubit_operations ??
      ops.filter((o) => o.qubits?.length === 2).length;
    return `<div class="program-metrics" data-stage="${id}">${[
      [
        ops.filter(
          (o) =>
            o.qubits?.length === 1 && !["measure", "reset"].includes(o.name),
        ).length,
        "1Q",
      ],
      [two, "2Q"],
      [m.depth ?? 0, "D"],
    ]
      .map(
        ([v, l]) =>
          `<div data-morph="metric-${id}-${l.replaceAll(" ", "-")}"><b>${fmt(v, 0)}</b><span>${l}</span></div>`,
      )
      .join("")}</div>`;
  }
  // Display excerpts without modifying the captured artifact or its line numbers.
  function fullCode(
    a,
    title,
    {
      columns = 1,
      height = 510,
      width = 800,
      hot = null,
      entryOnly = false,
    } = {},
  ) {
    if (!a?.code) return code(a, { title });
    const original = a.code.split("\n");
    const end = entryOnly
      ? original.findIndex(
          (l, i) => l === "}" && original[i - 1]?.includes("ret i64 0"),
        )
      : -1;
    const lines = [];
    original
      .slice(0, end < 0 ? original.length : end + 1)
      .forEach((raw, index) => {
        if (!raw.trim() && (entryOnly || index === original.length - 1)) return;
        lines.push({ raw, index });
      });
    const rows = Math.ceil(lines.length / columns);
    const longest = Math.max(...lines.map((l) => l.raw.length), 1);
    const font = Math.min(
      23,
      height / rows / 1.18,
      (width / columns - 40) /
        (entryOnly ? Math.min(longest, 80) : longest) /
        0.61,
    );
    const effect = (l) =>
      /reset/.test(l)
        ? "reset"
        : /phi |add |icmp.*slt|for |while |loop/.test(l)
          ? "loop"
          : /if |br |read_result|measure|__mz__/.test(l)
            ? "branch"
            : "gate";
    const parts = Array.from(
      { length: columns },
      (_, c) =>
        `<pre class="highlight">${lines
          .slice(c * rows, (c + 1) * rows)
          .map(
            ({ raw, index }) =>
              `<span class="code-line ${hot && effect(raw) === hot ? "hot" : ""}" data-effect="${effect(raw)}"><span class="line-no">${index + 1}</span><span class="code-text">${a.lines_html?.[index] ?? esc(raw)}</span></span>`,
          )
          .join("")}</pre>`,
    ).join("");
    return `<div class="code-frame full-code" data-morph="full-${esc(title)}" data-complete-lines="${original.length}"><div class="code-title">${esc(title)}</div><div class="complete-code" style="--code-columns:${columns};--code-font:${font}px">${parts}</div></div>`;
  }
  const evidence = data.compiler_evidence || {};
  const totalTime = (capture = run) => {
    const end = Math.max(
      0,
      ...(capture.events || []).map((e) => e.time_ms + (e.duration_ms || 0)),
      ...(capture.lifecycle || []).map((e) => e.time_ms),
    );
    return capture.program_completions
      ? end
      : Math.max(end, capture.completed_ms || 0, capture.duration_ms || 0);
  };
  const qdmiStart = (capture) =>
    capture.submission_started_ms ??
    capture.events?.find((e) => /try_submit_job|job_submit/.test(e.operation))
      ?.time_ms ??
    0;
  const interpolateTime = (capture, progress) =>
    qdmiStart(capture) + progress * (totalTime(capture) - qdmiStart(capture));
  const icon = (name, cls = "pictogram") =>
    `<span class="${cls}">${svg(viz.icon(name, 0, 0, 100), "0 0 100 100", name)}</span>`;
  const badge = (brand, label) =>
    `<div class="product-badge">${logo(brand)}<span>${label}</span></div>`;
  function runtimeSequence(capture, elapsed) {
    const adapter =
      capture.trace_kind === "python-adapter-and-native-program-completions";
    const events = [
      ...(capture.events || []),
      ...(capture.runtime_events || []),
      ...(capture.lifecycle || []),
    ].sort((a, b) => a.time_ms - b.time_ms);
    const first = (predicate) => events.find(predicate);
    const links = adapter
      ? [
          [
            0,
            1,
            "Compile OpenQASM → QIR",
            first((e) => e.state === "compiling"),
          ],
          [
            1,
            2,
            "Submit One Program Batch",
            first((e) => /try_submit_job/.test(e.operation)),
          ],
          [2, 2, "Execute Programs", first((e) => e.state === "running")],
          [
            2,
            1,
            "Retrieve Indexed Results",
            first((e) => Number.isInteger(e.program_index)),
          ],
          [1, 0, "Return Samples", first((e) => e.state === "done")],
        ]
      : [
          [
            0,
            1,
            "Submit Adaptive QIR",
            first((e) => /job_submit/.test(e.operation)),
          ],
          [
            1,
            2,
            "Compile and Initialize JIT",
            first((e) => e.operation === "JIT compilation begins"),
          ],
          [
            2,
            2,
            "Execute Adaptive Shots",
            first((e) => e.operation.startsWith("Adaptive sampling begins")),
          ],
          [
            1,
            0,
            "Retrieve Results",
            first((e) => /get_results/.test(e.operation)),
          ],
        ];
    const names = adapter
        ? ["Application", "Core / QDMI", "DDSIM"]
        : ["QDMI", "DDSIM", "LLVM JIT"],
      xs = [82, 320, 558];
    let body = names
      .map(
        (n, i) =>
          text(xs[i], 30, n, adapter ? 24 : 27) +
          line(xs[i], 53, xs[i], adapter ? 430 : 275, "#cbdced", 2),
      )
      .join("");
    links.forEach(([a, b, label, e], i) => {
      const on = !!e && elapsed >= e.time_ms,
        y = adapter ? 105 + i * 72 : 90 + i * 50;
      body += `<g data-morph="sequence-${i}" opacity="${on ? 1 : 0.12}">${a === b ? `<path d="M${xs[a]} ${y - 10}h34v26h-34" fill="none" stroke="#2f70b8" stroke-width="3"/>` : line(xs[a], y, xs[b], y)}${a !== b ? `<path d="M${xs[b] + (b > a ? -7 : 7)} ${y - 5}L${xs[b]} ${y}L${xs[b] + (b > a ? -7 : 7)} ${y + 5}" fill="none" stroke="#2f70b8" stroke-width="2"/>` : ""}${text(a === b ? xs[a] - 20 : (xs[a] + xs[b]) / 2, a === b ? y + 7 : y - 17, label, adapter ? 18 : 22, "#142b45", a === b ? "end" : "middle")}</g>`;
    });
    return `<div class="sequence-svg">${svg(body, adapter ? "0 0 640 450" : "0 0 640 290", "Timestamped QDMI workflow")}</div>`;
  }
  function energyChart(progress = 1, compare = true, elapsed = null) {
    const p = app.propagation;
    if (!p) return "";
    const curves = p.curves,
      ref = app.chemistry.fci_energy;
    const values = curves
      .flatMap((c) =>
        c.energy.flatMap((e, i) => [
          e - c.stderr[i] - ref,
          e + c.stderr[i] - ref,
        ]),
      )
      .map((e) => e * 1000);
    const padding = Math.max(
      1,
      (Math.max(...values) - Math.min(...values)) * 0.08,
    );
    const low = Math.min(0, ...values) - padding,
      high = Math.max(Math.abs(low) * 1.8, ...values) + padding;
    const x = (t) => 90 + (t / p.tau.at(-1)) * 640,
      y = (e) => 305 - ((e - low) / (high - low)) * 240;
    let body = text(
      90,
      30,
      "Energy − exact reference (mHa)",
      25,
      "#142b45",
      "start",
    );
    for (let i = 0; i <= 4; i++) {
      const value = low + ((high - low) * i) / 4,
        yy = y(value);
      body +=
        line(90, yy, 730, yy, "#e1eaf2", 1) +
        (Math.abs(yy - y(0)) > 28
          ? text(77, yy + 7, fmt(value, 0), 21, "#5b7088", "end")
          : "");
      body += text(
        90 + i * 160,
        337,
        fmt((p.tau.at(-1) * i) / 4, 1),
        21,
        "#5b7088",
      );
    }
    body +=
      line(90, y(0), 730, y(0), "#d6882f", 2) +
      text(77, y(0) + 7, "0", 21, "#a96916", "end") +
      text(405, 378, "Imaginary Time τ (Ha⁻¹)", 25);
    curves.forEach((c, j) => {
      if (j && !compare) return;
      const position =
        elapsed === null
          ? Math.max(
              0,
              Math.min(p.tau.length - 1, progress * (p.tau.length - 1)),
            )
          : recordedPosition(c.step_completed_ms || [], elapsed);
      const count =
        elapsed === null
          ? Math.floor(position) + 1
          : (c.step_completed_ms || []).filter((time) => time <= elapsed)
              .length;
      const coords = (sign = 0) => {
        const points = c.energy
          .slice(0, count)
          .map(
            (e, i) =>
              `${x(p.tau[i])},${y((e - ref + sign * c.stderr[i]) * 1000)}`,
          );
        if (count > 0 && count < p.tau.length) {
          const low = Math.floor(position),
            high = Math.min(low + 1, p.tau.length - 1),
            fraction = position - low,
            energy =
              c.energy[low] + (c.energy[high] - c.energy[low]) * fraction,
            error = c.stderr[low] + (c.stderr[high] - c.stderr[low]) * fraction,
            tau = p.tau[low] + (p.tau[high] - p.tau[low]) * fraction;
          points.push(`${x(tau)},${y((energy - ref + sign * error) * 1000)}`);
        }
        return points;
      };
      if (!j)
        body += `<polygon points="${[...coords(1), ...coords(-1).reverse()].join(" ")}" fill="#2f70b8" opacity=".12"/>`;
      body += `<polyline data-morph="energy-${j}" data-points="${count}" points="${coords().join(" ")}" stroke="${j ? "#5b7088" : "#2f70b8"}" stroke-width="${j ? 3 : 4}" fill="none" ${j ? 'stroke-dasharray="8 5"' : ""}/>`;
    });
    body +=
      line(80, 415, 125, 415, "#2f70b8", 4) +
      text(140, 423, "Quantum-Assisted Trial", 24, "#2f70b8", "start") +
      (compare
        ? `<path d="M475 415h45" stroke="#5b7088" stroke-width="3" stroke-dasharray="8 5"/>` +
          text(535, 423, "Hartree–Fock Trial", 24, "#5b7088", "start")
        : "");
    return `<div class="energy-chart">${svg(body, "0 0 820 440", "Measured energy convergence, both full time ranges retained")}</div>`;
  }
  function batchState(elapsed) {
    const capture = app.execution || {};
    const complete = (rows) =>
      new Set(
        (rows || [])
          .filter((e) => e.time_ms + (e.duration_ms || 0) <= elapsed)
          .map((e) => e.program_index),
      );
    return {
      prepared: complete(capture.preparation),
      compiled: complete(capture.compilation),
      done: complete(capture.program_completions),
      retrieved: complete(
        capture.retrievals ||
          capture.events?.filter((e) => Number.isInteger(e.program_index)),
      ),
    };
  }
  function batchMatrix(elapsed, preparing = false) {
    const { prepared, compiled, done, retrieved } = batchState(elapsed),
      n = app.workload?.snapshots || 0;
    // One cell per actual program; group no unrelated events into synthetic jobs.
    const cols = preparing ? 128 : 64,
      cell = 9,
      gap = 2,
      rows = Math.ceil(n / cols);
    const paths = ["", "", "", ""];
    for (let i = 0; i < n; i++) {
      const color = preparing
        ? prepared.has(i)
          ? 3
          : 0
        : retrieved.has(i)
          ? 3
          : done.has(i)
            ? 2
            : compiled.has(i)
              ? 1
              : 0;
      paths[color] +=
        `M${(i % cols) * (cell + gap)} ${Math.floor(i / cols) * (cell + gap)}h${cell}v${cell}h-${cell}z`;
    }
    // Each program retains its own cell; four paths avoid thousands of DOM updates per frame.
    const body = paths
      .map(
        (path, i) =>
          `<path d="${path}" fill="${["#edf2f7", "#bbd1e8", "#d6882f", "#2f70b8"][i]}"/>`,
      )
      .join("");
    return `<div class="batch-grid ${preparing ? "preparation-grid" : ""}" data-morph="${preparing ? "preparation" : "execution"}-grid" data-prepared="${prepared.size}" data-programs="${retrieved.size}" data-completed="${done.size}" data-compiled="${compiled.size}" data-elapsed-ms="${elapsed}">${svg(body, `0 0 ${cols * (cell + gap)} ${rows * (cell + gap)}`, preparing ? "One cell per prepared OpenQASM program" : "One cell per program: compiled, completed, retrieved")}<div class="batch-counts">${preparing ? `<span>${fmt(prepared.size, 0)} / ${fmt(n, 0)} Programs Prepared</span>` : `<span>${fmt(compiled.size, 0)} Compiled</span><span>${fmt(done.size, 0)} Completed</span><span>${fmt(retrieved.size, 0)} Retrieved</span>`}</div></div>`;
  }
  function shadowBatchDrawing() {
    let body = [2, 1, 0]
      .map(
        (i) =>
          `<rect x="${43 + i * 13}" y="${38 - i * 10}" width="400" height="174" rx="12" fill="white" stroke="#a9bfd4" stroke-width="2"/>`,
      )
      .join("");
    body += [0, 1, 2, 3]
      .map((i) => line(67, 69 + i * 35, 422, 69 + i * 35, "#a0b9d1", 2))
      .join("");
    body += `<rect x="76" y="54" width="51" height="137" rx="8" fill="#f1f5f9" stroke="#a0b9d1" stroke-width="2"/><rect x="141" y="54" width="93" height="137" rx="8" fill="#eaf2fa" stroke="#2f70b8" stroke-width="2"/><rect x="249" y="54" width="111" height="137" rx="8" fill="#fff4e5" stroke="#d6882f" stroke-width="2"/>${text(102, 112, "H", 20)}${text(102, 144, "CX", 20)}<text x="187" y="130" font-family="Inter,Arial,sans-serif" font-size="31" fill="#142b45" text-anchor="middle">U<tspan baseline-shift="sub" font-size="20">trial</tspan></text>${text(304, 133, "U(Qₖ)", 31, "#a96916")}`;
    body += [0, 1, 2, 3]
      .map(
        (i) =>
          `<rect x="375" y="${57 + i * 35}" width="29" height="25" rx="4" fill="white" stroke="#7fa8cb"/><path d="M381 ${75 + i * 35}a9 9 0 0 1 17 0M389 ${74 + i * 35}l6 -10" fill="none" stroke="#2f70b8" stroke-width="1.5"/>`,
      )
      .join("");
    body += text(265, 254, "One Random Basis Qₖ per Program", 24, "#5b7088");
    return body;
  }
  function compactShadowBatch(progress = 1) {
    const p = Math.max(0, Math.min(1, progress)),
      body = shadowBatchDrawing();
    return `<div class="compact-shadow-batch" style="opacity:${p}">${svg(body, "0 0 540 278", "Compact four-wire batch: fixed trial and vacuum reference, random basis U(Q_k), measurement")}</div>`;
  }
  function preparationSketch(progress = 1) {
    const p = Math.max(0, Math.min(1, progress));
    let body = `<defs><radialGradient id="molecule-orbital"><stop stop-color="#2f70b8" stop-opacity=".35"/><stop offset="1" stop-color="#2f70b8" stop-opacity="0"/></radialGradient></defs><ellipse cx="110" cy="76" rx="97" ry="58" fill="url(#molecule-orbital)"/><ellipse cx="174" cy="76" rx="97" ry="58" fill="url(#molecule-orbital)"/>${line(106, 76, 178, 76, "#6fa8d8", 8)}<circle cx="106" cy="76" r="21" fill="#2f70b8"/><circle cx="178" cy="76" r="21" fill="#62aaca"/>${text(141, 144, "Molecular Hamiltonian", 25)}`;
    body += `<g opacity="${Math.min(1, p * 3)}"><path d="M145 160V195" stroke="#86aaca" stroke-width="3"/><rect x="25" y="207" width="250" height="90" rx="13" fill="#eaf2fa" stroke="#a9c6e1"/><text x="150" y="244" font-family="Inter,Arial,sans-serif" font-size="25" fill="#142b45" text-anchor="middle">|Ψ<tspan baseline-shift="sub" font-size="17">trial</tspan>⟩ = U<tspan baseline-shift="sub" font-size="17">trial</tspan>|HF⟩</text>${text(150, 279, "Fixed Trial State", 22, "#5b7088")}</g>`;
    body += `<g opacity="${Math.max(0, Math.min(1, p * 3 - 1))}">${line(286, 250, 316, 250)}<rect x="329" y="207" width="186" height="90" rx="13" fill="#fff4e5" stroke="#dfb579"/>${text(422, 245, "U(Qₖ)", 30, "#a96916")}${text(422, 279, "Random Basis k", 22)}</g>`;
    body += `<g opacity="${Math.max(0, Math.min(1, p * 3 - 2))}">${line(422, 297, 422, 358)}<g transform="translate(0 340)">${shadowBatchDrawing()}</g></g>`;
    return `<div class="preparation-sketch">${svg(body, "0 0 540 618", "Molecule and trial-state preparation connected through a random measurement basis to the circuit batch")}</div>`;
  }
  function sampleStream(progress = 1) {
    const preview = app.shadow_preview || [],
      shown = Math.ceil(progress * preview.length);
    return `<div class="sample-stream">${preview
      .slice(0, shown)
      .map(
        (sample, i) =>
          `<div data-morph="shadow-sample-${i}" style="opacity:${Math.min(1, progress * preview.length - i)}"><small>Program ${sample.program_index}</small><strong>${(sample.shots || []).map(esc).join("<br>")}<span>⋮</span></strong></div>`,
      )
      .join(
        "",
      )}<span class="sample-continuation" style="opacity:${progress}" aria-label="More program outputs">…</span></div>`;
  }
  const appHeading = "Quantum-Assisted Auxiliary-Field Quantum Monte Carlo";
  const applicationHeading = () =>
    `<h2 class="application-heading" data-morph="heading">${appHeading}<small>QC-AFQMC</small></h2>`;
  const appReference = () =>
    `<div class="application-paper-reference" data-morph="application-paper-reference">${qr("afqmc-paper", "arxiv.org/abs/2506.22408", "Quantum-Assisted Auxiliary-Field Quantum Monte Carlo")}</div><div class="application-reference" data-morph="application-reference">${logo("aws")}${qr("afqmc", "github.com/amazon-braket/amazon-braket-examples", "Amazon Braket · Public AFQMC Example", false)}</div>`;
  function appPanel(i, body) {
    return `<section data-morph="app-pillar-${i}" class="app-panel"><h3>${icon(["workstation", "cryostat", "server"][i])}<span>${["Classical Preparation", "Quantum Shadow Collection", "Classical Monte Carlo"][i]}</span></h3><div class="app-panel-body">${body}</div><div class="app-badges">${i === 0 ? badge("pennylane", "") : i === 1 ? badge("mqt", "Compiler Collection") + badge("qdmi", "") : badge("mqsc", "Orchestrator")}</div></section>`;
  }
  function appBody(i, progress = 1) {
    if (i === 0) return preparationSketch(progress);
    if (i === 1)
      return `<p class="panel-kicker">${fmt(app.workload?.snapshots, 0)} Circuits × ${fmt(app.workload?.shots_per_snapshot, 0)} Shots</p>${compactShadowBatch()}${sampleStream(progress)}`;
    return `<p class="panel-kicker">Propagate → Overlap → Reweight → Reduce</p>${walkerScene(progress)}<div class="equation-caption"><img src="${assets["eq-afqmc-mean"]}" alt="Weighted local energy"/><span>Weighted Mean</span></div>${energyChart(progress)}`;
  }
  function applicationScene(step, progress = 1) {
    return `<div class="app-panels intro">${[0, 1, 2].map((i) => `<div class="${i > step ? "future" : ""}">${appPanel(i, appBody(i, i === step ? progress : 1))}</div>`).join("")}</div>${appReference()}`;
  }
  function preparationTrace(elapsed) {
    const rows = app.execution?.preprocessing || [],
      first = app.execution?.preparation?.[0]?.time_ms || 0,
      conversion = {
        id: "payloads",
        label: "Convert each tape to OpenQASM",
        code: "program = original_convert(tape)",
        time_ms: first,
        duration_ms: phaseTime("compiling") - first,
      };
    return `<div class="preprocessing-trace" data-morph="preprocessing-trace">${[
      ...rows,
      conversion,
    ]
      .map((row) => {
        const end = row.time_ms + row.duration_ms,
          active = elapsed >= row.time_ms && elapsed < end,
          done = elapsed >= end;
        return `<div data-morph="preprocessing-${row.id}" data-start-ms="${row.time_ms}" data-end-ms="${end}" class="${active ? "active" : done ? "done" : "pending"}"><div><b>${esc(row.label)}</b><time>${fmt(row.time_ms / 1000, 2)}–${fmt(end / 1000, 2)} s</time></div><code title="${esc(row.code)}">${esc(row.code.split("\n")[0])}</code></div>`;
      })
      .join("")}</div>`;
  }
  function preparationCode(elapsed, trace = false) {
    const definitions = app.source_definitions || {};
    // The build supplies verbatim executable lines with docstrings omitted for display.
    const left =
        app.presentation_source?.left ||
        [definitions.hydrogen_trial, definitions.apply_gaussian_givens]
          .filter(Boolean)
          .join("\n"),
      right =
        app.presentation_source?.right ||
        [definitions.apply_pauli_layer, app.source, app.batch_source]
          .filter(Boolean)
          .join("\n");
    return `<div class="preparation-code">${[left, right].map((source, i) => `<div>${trace && !i ? preparationTrace(elapsed) : fullCode({ code: source, lines_html: app.presentation_source?.[i ? "right_lines_html" : "left_lines_html"] }, i ? "Shadow Circuit and Batch" : "Trial State and Gaussian Rotations", { height: i ? 400 : 255, width: 825 })}${i ? "" : batchMatrix(elapsed, true)}</div>`).join("")}</div>`;
  }
  const phaseTime = (name) =>
    app.execution?.lifecycle?.find((e) => e.state === name)?.time_ms || 0;
  function preparationTime(progress) {
    const phases = app.execution?.preprocessing || [],
      first = app.execution?.preparation?.[0]?.time_ms || 0,
      ends = phases.length
        ? [0, ...phases.map((phase) => phase.time_ms + phase.duration_ms)]
        : [0, first];
    if (progress >= 0.7)
      return (
        first + ((phaseTime("compiling") - first) * (progress - 0.7)) / 0.3
      );
    const scaled = (progress / 0.7) * (ends.length - 1),
      index = Math.min(ends.length - 2, Math.floor(scaled));
    return ends[index] + (ends[index + 1] - ends[index]) * (scaled - index);
  }
  const quantumPlayback = {
    duration: 4000 + 13000 / 1.5 + 3000,
    transition: 850,
  };
  function quantumTime(progress) {
    const capture = app.execution;
    const breaks = [
      phaseTime("compiling"),
      qdmiStart(capture),
      capture.retrievals?.[0]?.time_ms || phaseTime("extracting-results"),
      totalTime(capture),
    ];
    const stops = [
      0,
      4000 / quantumPlayback.duration,
      1 - 3000 / quantumPlayback.duration,
      1,
    ];
    const segment = progress < stops[1] ? 0 : progress < stops[2] ? 1 : 2;
    const fraction =
      (progress - stops[segment]) / (stops[segment + 1] - stops[segment]);
    return breaks[segment] + fraction * (breaks[segment + 1] - breaks[segment]);
  }
  function propagationTime(progress) {
    const start = totalTime(app.execution),
      end = app.timeline?.completed_ms || start,
      origin = app.timeline?.propagation_started_ms || start,
      curves = app.propagation?.curves || [],
      q = curves[0]?.step_completed_ms || [0],
      hf = curves[1]?.step_completed_ms || q,
      observed = [
        start,
        origin + q[0],
        origin + q.at(-1),
        origin + hf[0],
        origin + hf.at(-1),
        end,
      ],
      // Actual windows may overlap; monotone clamping never rewinds the application clock.
      breaks = observed.map((time, i) => Math.max(...observed.slice(0, i + 1))),
      stops = [0, 0.12, 0.52, 0.6, 0.94, 1],
      next = stops.findIndex((stop) => stop > progress),
      segment = next < 0 ? stops.length - 2 : Math.max(0, next - 1),
      fraction =
        (progress - stops[segment]) / (stops[segment + 1] - stops[segment]);
    return breaks[segment] + fraction * (breaks[segment + 1] - breaks[segment]);
  }
  function workflowClock(step, progress = 1) {
    const finish = totalTime(app.execution),
      complete = app.timeline?.completed_ms || finish;

    const elapsed =
      step === 0
        ? 0
        : step === 1
          ? preparationTime(progress)
          : step === 2
            ? quantumTime(progress)
            : step === 3
              ? propagationTime(progress)
              : complete;
    const clock = `${fmt(elapsed / 1000, 2)} s · Recorded Application Clock`;
    return `<div class="workflow-clock" data-morph="workflow-clock" data-elapsed-ms="${elapsed}"><div class="clock-stages"><span class="${step <= 1 ? "active" : ""}">Prepare</span><span class="${step === 2 ? "active" : ""}">Compile · Run · Retrieve</span><span class="${step >= 3 ? "active" : ""}">Propagate on Slurm</span><b>${clock}</b></div><div class="clock-line"><div style="width:${(100 * elapsed) / Math.max(1, complete)}%"></div></div></div>`;
  }
  function hybridExecution(step, progress = 1) {
    const focused = step < 2 ? 0 : step === 2 ? 1 : step === 3 ? 2 : -1;
    const elapsed =
      step === 2 ? quantumTime(progress) : step > 2 ? Infinity : 0;
    const quantum = `<div class="quantum-batch"><div>${runtimeSequence(app.execution, elapsed)}<p class="panel-kicker">One QDMI Job · ${fmt(app.workload?.snapshots, 0)} Programs</p></div>${batchMatrix(elapsed)}</div>`;
    const workers = app.propagation?.workers || [],
      nodes = new Set(workers.map((w) => w.node)).size;
    const propagationElapsed =
      step === 3
        ? propagationTime(progress) -
          (app.timeline?.propagation_started_ms || Infinity)
        : Infinity;
    const classical = `<div class="slurm-allocation">${logo("slurm")}<span>${nodes} Compute Nodes · ${workers.length} Slurm Ranks</span></div><div class="walker-comparison">${[0, 1].map((trial) => `<div><b>${trial ? "Hartree–Fock Reference" : "Quantum-Assisted Trial"}</b>${walkerScene(1, propagationElapsed, trial)}</div>`).join("")}</div>${energyChart(1, true, propagationElapsed)}`;
    const bodies = [
      step < 2
        ? preparationCode(
            step === 0 ? 0 : preparationTime(progress),
            step === 1,
          )
        : appBody(0),
      step >= 2 ? quantum : appBody(1),
      step >= 3 ? classical : appBody(2),
    ];
    return `<div class="app-panels execution focus-${focused}">${bodies.map((body, i) => appPanel(i, body)).join("")}</div>${workflowClock(step, progress)}<div class="workflow-foot" data-morph="workflow-foot"><span>${fmt(app.workload?.snapshots, 0)} Programs · ${fmt(app.workload?.total_shots, 0)} Shots · ${app.propagation?.walkers} Walkers per Trial</span><span>Ideal DDSIM · ${app.execution?.simulator_threads ?? "Recorded"} Executor Slots · Time-Scaled Replay</span></div>`;
  }
  const compilerLabels = [
    "Input Program",
    "Optimization",
    "Device Specification",
    "Placement & Routing",
    "Native Synthesis",
    "Executable",
  ];
  function compilerFlow(active, content, { complete = false } = {}) {
    return `<div class="compiler-tiles">${compilerLabels.map((label, i) => `<div data-morph="compile-tile-${i}" class="compile-tile ${active === i ? "selected" : ""} ${complete || i < active ? "complete" : ""}"><span>${i + 1}</span><b>${label}</b></div>`).join("")}</div>${content}`;
  }
  function deviceCard(
    target,
    { layout = null, activeSites = [], blend = null } = {},
  ) {
    const i = targets.indexOf(target),
      brand = ["iqm", "ibm", "aws"][i],
      repository = [
        "iqm-finland/QDMI-on-IQM",
        "munich-quantum-software/ibm-qdmi-device",
        "munich-quantum-software/amazon-braket-qdmi-device",
      ][i],
      reference = ["iqm", "ibm", "braket"][i];
    return `<div class="spec-card" data-morph="device-card" data-target="${esc(target.id)}"><div class="spec-step"><b>3 · Device Specification</b><span>Powered by ${logo("qdmi")}</span></div><div class="spec-brand">${logo(brand)}</div>${deviceView(target, activeSites, layout, blend)}<div class="spec-details"><b>${["Emerald", "Nighthawk · ibm_miami", "IonQ Forte-1"][i]}</b><span>${target.metadata.qubits} Qubits</span><span>${["QIR · IQM-JSON", "QPY", "OpenQASM 3"][i]}</span><span>${["R · CZ", "RZ · SX · X · CZ", "GPI · GPI2 · RZZ"][i]}</span></div><a class="spec-repository" href="https://github.com/${repository}"><img src="${assets[`qr-${reference}`]}" alt="QDMI implementation QR code"/><span>${logo("github")}<span>${repository.replace("/", "/<wbr>")}</span></span></a></div>`;
  }
  function routingFrame(target, progress) {
    const sweep = target.compilation.routing_trace?.find(
      (e) => e.phase === "final-routing",
    );
    const initial = target.compilation.layout?.initial || [];
    if (!sweep?.steps?.length)
      return {
        c: targetStage(target, "place-and-route")?.circuit,
        layout: initial,
        active: -1,
        sites: [],
        swaps: 0,
      };
    const steps = sweep.steps,
      weights = steps.map((e) => (e.operation === "qco.swap" ? 5 : 1)),
      total = weights.reduce((a, b) => a + b, 0);
    let consumed = 0,
      index = steps.length - 1,
      fraction = 1;
    for (let i = 0; i < steps.length; i++) {
      if (consumed + weights[i] > progress * total) {
        index = i;
        fraction = (progress * total - consumed) / weights[i];
        break;
      }
      consumed += weights[i];
    }
    let layout = sweep.before.slice(0, initial.length),
      before = [...layout],
      swaps = 0;
    steps.slice(0, index + 1).forEach((e, i) => {
      if (i === index) before = [...layout];
      if (e.operation === "qco.swap") {
        const [a, b] = e.sites;
        layout = layout.map((p) => (p === a ? b : p === b ? a : p));
        swaps++;
      }
    });
    const sites = [...new Set(steps.flatMap((e) => e.sites))].sort(
      (a, b) => a - b,
    );
    const c = {
      qubits: sites.map((id) => ({ id, site: id, label: `${id}` })),
      operations: steps.map((e, i) => ({
        id: `route-${i}`,
        name: e.controls
          ? "c".repeat(e.controls) + (e.base_gate || "").replace(/^qco\./, "")
          : (e.base_gate || e.operation).replace(/^qco\./, ""),
        qubits: e.sites,
        parameters: e.parameters || [],
        num_controls: e.controls || 0,
      })),
    };
    return {
      c,
      layout,
      active: progress >= 1 ? -1 : index,
      reveal: progress >= 1 ? Infinity : index + 1,
      sites: progress >= 1 ? [] : steps[index].sites,
      blend: { before, fraction: progress >= 1 ? 1 : fraction },
      swaps,
    };
  }
  function stageCard(target, id, title, options = {}) {
    return `<section class="compiled-stage stage-${id}" data-target="${esc(target.id)}"><div class="stage-shell" data-morph="stage-${id}" aria-hidden="true"></div><h3 data-morph="stage-title-${id}">${title}</h3>${circuit(options.c || targetStage(target, id)?.circuit, { full: true, key: id, ...options })}${options.metrics === false ? "" : programMetrics(target, id)}</section>`;
  }
  function compilerSummary() {
    const stacks = [
      [
        "Frontends",
        `<span>${logo("qiskit")}<b>Qiskit</b></span><b>OpenQASM 3</b>${logo("pennylane")}`,
        "Frontend",
        "frontends",
      ],
      [
        "Our Ecosystem",
        `<div class="ecosystem-group"><h4>Quantum Hardware &amp; Cloud</h4><div class="ecosystem-logos hardware">${["iqm", "ibm", "aqt", "planqc", "wmi", "peak-quantum", "aws"].map((name) => logo(name)).join("")}</div></div><div class="ecosystem-group"><h4>Simulators</h4><div class="ecosystem-logos simulators"><span>${logo("mqt")}<b>DDSIM</b></span><span>${logo("eviden")}<b>Qaptiva</b></span></div></div><div class="ecosystem-group"><h4>HPC Centers</h4><div class="ecosystem-logos centers">${["lrz", "psnc", "vtt", "ornl", "cineca", "cesga"].map((name) => logo(name)).join("")}</div></div><div class="ecosystem-group"><h4>Integration Projects</h4><div class="ecosystem-logos initiatives">${["mqss", "fullstaqd", "eqs3"].map((name) => logo(name)).join("")}<span>${logo("openqse")}<b>openQSE</b></span></div></div>`,
        "Integration",
        "integrations",
      ],
      [
        "Outputs",
        `${logo("qir")}<b>OpenQASM 3</b><span>${logo("qiskit")}<b>Qiskit</b></span>`,
        "Output Format",
        "outputs",
      ],
    ];
    return `<div class="compiler-ecosystem">${stacks.map(([title, body, next, kind], i) => `<section class="ecosystem-${kind}" data-morph="ecosystem-${i}"><h3>${title}</h3><div class="ecosystem-body">${body}</div><p>Your <b>${next}</b><br>Could Be Next</p></section>`).join("")}</div>`;
  }
  function compilationScene(step, progress = 1) {
    if (step === 9)
      return compilerFlow(5, compilerSummary(), { complete: true });
    const targetIndex = step === 8 ? 2 : step === 7 ? 1 : 0,
      target = targets[targetIndex],
      stage = Math.min(step, 6),
      route =
        stage >= 3
          ? routingFrame(target, stage === 3 ? 0 : stage === 4 ? progress : 1)
          : null;
    const input = stageCard(targets[0], "source", "1 · Input Program"),
      optimized =
        stage >= 1
          ? stageCard(targets[0], "optimized", "2 · Optimization")
          : "";
    let routed = "",
      native = "",
      executable = "";
    if (route)
      routed = stageCard(target, "place-and-route", "4 · Placement & Routing", {
        c: route.c,
        active: stage === 3 ? -1 : route.active,
        reveal:
          stage === 3 ? 0 : stage === 4 && progress === 0 ? 0 : route.reveal,
        layout: target.compilation.layout.initial,
        metrics: stage > 4 || (stage === 4 && progress >= 1),
      });
    if (stage >= 5)
      native = stageCard(
        target,
        "target-native-synthesis",
        "5 · Native Synthesis",
        {
          layout: target.compilation.layout.initial,
          fold: stage >= 6 ? [12, 22, 20][targetIndex] : null,
        },
      );
    if (stage >= 6) {
      const format = ["qir-adaptive", "qpy", "openqasm3"][targetIndex];
      const artifact = targetStage(target, format);
      const output =
        format === "qpy"
          ? `<h3>6 · Qiskit Circuit → program.qpy</h3><img class="qiskit-output" src="data:image/svg+xml;charset=utf-8,${encodeURIComponent(artifact.svg)}" alt="Qiskit drawing of the compiled IBM program"/>${code(artifact, { count: 5, key: "compiler-output" })}`
          : `${code(artifact, { start: format === "qir-adaptive" ? focus(artifact, "define", 0) : 0, count: format === "qir-adaptive" ? 13 : 16, title: format === "qir-adaptive" ? "6 · QIR 2.1 Adaptive Profile" : "6 · OpenQASM 3", key: "compiler-output" })}<p class="code-continuation">⋮ Continued in the complete file</p>`;
      executable = `<section class="compiler-output output-${format}" data-morph="compiler-output" data-target="${esc(target.id)}" data-format="${format}">${output}</section>`;
    }
    return compilerFlow(
      stage <= 2 ? stage : stage <= 4 ? 3 : stage === 5 ? 4 : 5,
      `<div class="compiler-workbench phase-${stage} ${stage >= 6 ? "with-output" : ""}"><div class="compiler-invariants">${input}${optimized}</div><div class="compiler-device" data-morph="compiler-device">${stage >= 2 ? deviceCard(target, { layout: route?.layout, activeSites: stage === 4 ? route?.sites : [], blend: stage === 4 ? route?.blend : null }) : ""}</div><div class="compiler-stages">${routed}${native}</div>${executable}</div>`,
    );
  }
  function measuredGrowth(progress = 1) {
    const points = evidence.grover_size?.points || [];
    if (!points.length)
      return '<p class="note">Grover measurements are being regenerated.</p>';
    const x0 = 160,
      y0 = 530,
      w = 1300,
      h = 420,
      minQ = points[0].qubits,
      maxQ = points.at(-1).qubits;
    const max = Math.ceil(
      Math.log10(Math.max(...points.map((p) => p.unrolled_bytes))),
    );
    const min = Math.floor(
      Math.log10(Math.min(...points.map((p) => p.preserved_bytes))),
    );
    const x = (q) => x0 + ((q - minQ) / (maxQ - minQ)) * w,
      y = (v) => y0 - ((Math.log10(v) - min) / (max - min)) * h;
    let body = text(
      160,
      45,
      "Emitted program size · bytes · logarithmic scale",
      30,
      "#142b45",
      "start",
    );
    for (let power = min; power <= max; power++)
      body +=
        line(x0, y(10 ** power), x0 + w, y(10 ** power), "#dbe5ef", 1) +
        text(
          x0 - 24,
          y(10 ** power) + 8,
          `10${["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸"][power] || "^" + power}`,
          26,
          "#526d88",
          "end",
        );
    for (const [key, color, label] of [
      ["unrolled_bytes", "#cc704b", "Fully unrolled"],
      ["preserved_bytes", "#2f70b8", "Structure preserved"],
    ]) {
      const unrolled = key === "unrolled_bytes";
      const fraction = unrolled
        ? Math.min(1, progress * 2)
        : Math.max(0, progress * 2 - 1);
      const shown = points.slice(0, Math.ceil(fraction * points.length));
      const dash = unrolled ? 'stroke-dasharray="14 9"' : "";
      body += `<polyline points="${shown.map((p) => `${x(p.qubits)},${y(p[key])}`).join(" ")}" fill="none" stroke="${color}" stroke-width="5" ${dash}/>`;
      body += shown
        .map((p) =>
          unrolled
            ? `<rect x="${x(p.qubits) - 6}" y="${y(p[key]) - 6}" width="12" height="12" fill="${color}"/>`
            : `<circle cx="${x(p.qubits)}" cy="${y(p[key])}" r="6" fill="${color}"/>`,
        )
        .join("");
      const legendY = unrolled ? 42 : 82;
      body +=
        `<path d="M990 ${legendY}h70" stroke="${color}" stroke-width="5" ${dash}/>` +
        (unrolled
          ? `<rect x="1019" y="${legendY - 6}" width="12" height="12" fill="${color}"/>`
          : `<circle cx="1025" cy="${legendY}" r="6" fill="${color}"/>`) +
        text(1075, legendY + 8, label, 30, color, "start");
    }
    points
      .filter((_, i) => i % 2 === 0)
      .forEach((p) => {
        body += text(x(p.qubits), 574, p.qubits, 27, "#526d88");
      });
    body += text(810, 626, "Number of qubits", 31);
    return `<div class="growth-chart">${svg(body, "0 0 1640 660", "Measured MQT Core Grover program sizes, preserved and fully unrolled")}</div>`;
  }
  function paperFooter() {
    return `<div class="structure-footer" data-morph="structure-footer"><div class="exchange-footer">${logo("mqt")}<span>⇄</span>${logo("jeff")}<span>⇄</span><div>Tools &amp; Frameworks</div></div><div class="paper-partners">${["tum-mark", "mqsc", "oxford", "quantinuum", "ubc", "xanadu", "unitary"].map((n) => logo(n)).join("")}</div>${qr("unrolling", "arxiv.org/abs/2609.16171", "Rovara, Haag, et al., 2026")}</div>`;
  }
  function structureScene(step) {
    const v = variant("parity"),
      source = v.stages.find((a) => a.id === "source"),
      qir = v.exports.find((a) => a.id === "qir-adaptive"),
      effect = [null, null, "reset", "branch", "loop"][step];
    const active = flatten(source.circuit).flatMap((o, i) =>
      (effect === "reset" && o.name === "reset") ||
      (effect === "branch" &&
        (o.name === "measure" || o.region.some((r) => r.name === "if_else"))) ||
      (effect === "loop" && o.region.some((r) => r.name === "for_loop"))
        ? [i]
        : [],
    );
    const start = effect
      ? focus(
          qir,
          effect === "reset"
            ? "__reset__"
            : effect === "branch"
              ? "__mz__"
              : "phi ",
          effect === "loop" ? 1 : 0,
        )
      : 0;
    return `<div class="structured-program"><div class="structured-source" data-morph="structured-source">${fullCode(source, "Syndrome Extraction", { height: 310, width: 560, hot: effect })}${circuit(source.circuit, { full: true, key: "syndrome", active, hotRegion: effect === "loop" ? "for_loop" : effect === "branch" ? "if_else" : null })}</div><div class="structured-qir ${step === 0 ? "future" : ""}" data-morph="structured-qir">${fullCode(qir, "QIR 2.1 Adaptive Profile · Complete Entry Point", { columns: 2, height: 340, width: 1140, hot: effect, entryOnly: true })}<div class="control-lens ${effect ? "" : "future"}" data-morph="control-lens">${code(qir, { start, count: effect === "loop" ? 7 : effect === "branch" ? 7 : 5, title: `Program Excerpt · ${effect === "reset" ? "Reset" : effect === "branch" ? "Conditionals" : "Loops"}`, compact: true, key: "structured-detail" })}</div></div></div>${paperFooter()}`;
  }
  function growthScene(progress = 1) {
    const sketch = `${line(75, 104, 590, 104, "#a9c2d8", 3)}${line(76, 125, 100, 84, "#526d88", 2)}${text(87, 68, "n", 25)}<rect x="120" y="64" width="80" height="81" rx="10" fill="#eaf2fa" stroke="#2f70b8"/><text x="160" y="115" text-anchor="middle" font-family="STIX Two Math,serif" font-size="27">H<tspan baseline-shift="super" font-size="70%">⊗n</tspan></text><rect x="229" y="27" width="323" height="158" rx="18" fill="#eaf2fa" stroke="#2f70b8" stroke-width="3"/><rect x="250" y="63" width="114" height="83" rx="8" fill="white" stroke="#2f70b8"/><rect x="389" y="63" width="140" height="83" rx="8" fill="white" stroke="#2f70b8"/>${text(307, 112, "Oracle", 24)}${text(459, 112, "Diffusion", 24)}<path d="M529 216H250m10 -8l-10 8 10 8" fill="none" stroke="#2f70b8" stroke-width="3"/>${text(390, 255, "Repeat k Times", 27)}`;
    return `<div class="grover-growth"><div><h3 class="grover-label">Grover Search</h3>${svg(sketch, "0 0 630 280", "Grover loop on n qubits, with no ancilla")}<img class="grover-equation" src="${assets["eq-grover-iterations"]}" alt="k equals floor of pi divided by four arcsine of 2 to the minus n over two"/></div>${measuredGrowth(progress)}</div>${paperFooter()}`;
  }
  function executionLog(capture, elapsed) {
    const events = [
      ...(capture.events || []),
      ...(capture.runtime_events || []),
    ].sort((a, b) => a.time_ms - b.time_ms);
    const rows = [];
    for (const event of events.filter(
      (e) =>
        !Number.isInteger(e.program_index) &&
        !/^MQT_DDSIM_QDMI_device_(?:initialize|finalize|session_(?:alloc|init|free))$/u.test(
          e.operation,
        ),
    )) {
      const previous = rows.at(-1);
      if (previous?.operation === event.operation) {
        previous.times.push(event.time_ms);
      } else rows.push({ ...event, times: [event.time_ms] });
    }
    return `<div class="execution-log" data-recorded-calls="${events.length}"><h3>Job + Runtime Calls</h3>${rows.map((e, i) => `<div data-morph="log-${i}" class="${elapsed < e.time_ms ? "future" : ""}"><time>${fmt(Math.max(0, e.time_ms) / 1000, 3)} s</time><code>${esc(e.operation.replace(/^MQT_DDSIM_QDMI_/u, "").replace(/\(.*$/u, "()"))}</code>${e.times?.length > 1 ? `<b>×${e.times.filter((time) => time <= elapsed).length}</b>` : ""}</div>`).join("")}</div>`;
  }
  function workerProgress(capture, elapsed) {
    const events = capture.worker_events || [];
    const ids = [...new Set(events.map((event) => event.worker_index))].sort(
      (a, b) => a - b,
    );
    return `<div class="worker-progress" data-morph="worker-progress"><h4>${ids.length} Parallel QIR Workers</h4>${ids
      .map((id) => {
        const worker = events.filter((event) => event.worker_index === id);
        const completed =
          worker.findLast((event) => event.time_ms <= elapsed)?.completed || 0;
        const total = worker.at(-1).total;
        return `<div class="worker-row" data-worker="${id}" data-completed="${completed}"><span>Core ${id + 1}</span><div class="worker-track"><i style="width:${(100 * completed) / total}%"></i></div><b>${fmt(completed, 0)} / ${fmt(total, 0)}</b></div>`;
      })
      .join("")}</div>`;
  }
  function adaptiveRuntime(step, progress = 1) {
    const elapsed =
        step === 0 ? qdmiStart(run) : interpolateTime(run, progress),
      source = variant("qpe").stages.find((a) => a.id === "source");
    return `<div class="adaptive-runtime"><div class="qpe-program"><h3>Iterative QPE · φ = 1/3 · 8 Bits</h3>${MQSF_CIRCUIT.renderQPE(source.circuit)}<div class="runtime-columns">${executionLog(run, elapsed)}<div class="adaptive-sequence"><h3>QDMI → Runtime</h3>${runtimeSequence(run, elapsed)}</div></div></div><div class="runtime-backend"><div class="runtime-products"><div>${logo("qdmi")}<b>Device Interface</b></div><span>→</span><div>${logo("mqt")}<b>DDSIM</b></div><span>→</span><div>${logo("llvm")}<b>JIT Runtime</b></div></div>${workerProgress(run, elapsed)}<div class="runtime-outcomes"><h3>Measurement Outcomes</h3>${histogram(run, elapsed)}</div></div></div><div class="runtime-clock" data-morph="runtime-clock"><span>${fmt((elapsed - qdmiStart(run)) / 1000, 3)} s Recorded Time</span><span>Adaptive QIR 2.1 · 64-Shot Completion Batches · Slowed Replay</span><div class="clock-line"><div style="width:${(100 * (elapsed - qdmiStart(run))) / (totalTime(run) - qdmiStart(run))}%"></div></div></div>`;
  }
  function featureOverview(step, progress = 1) {
    const groups = [
      [
        "circuit",
        "OpenQASM 3.1",
        "Modifiers · Reusable Gates · Classical Expressions",
        "if / switch / for / while · Loop-Indexed Qubits",
      ],
      [
        "measurement",
        "QIR 2.1",
        "Base + Adaptive Profiles",
        "Compile and Execute · Arrays",
      ],
      [
        "qiskit",
        "Qiskit 2.5",
        "Native C API Integration",
        "SamplerV2 · EstimatorV2 · QDMI Batches",
      ],
      [
        "pennylane",
        "PennyLane",
        "Hybrid Workflows Through QDMI",
        "Samples · Expectation Values · Batch Programs",
      ],
      [
        "chip",
        "Ready to Install",
        "Linux, macOS, Windows Stable ABI Wheels",
        "Python 3.11–3.15(t) · Portable LLVM Distributions",
        "Modern Tooling · uv / ruff / ty",
      ],
      [
        "developer",
        "Ready to Build On",
        "Tutorials · Guides · API Reference",
        "Contribution Guidelines · Agent Instructions",
      ],
    ];
    const survey =
      step === 5 ? Math.max(0, Math.min(1, (progress - 0.5) * 2)) : 0;
    const references =
      step === 5 ? 1 - Math.max(0, Math.min(1, (progress - 0.3) * 5)) : 1;
    return `<div class="feature-grid">${groups.map(([art, title, first, second, third], i) => `<section data-morph="feature-${i}" class="feature-card ${i > step ? "future" : ""}"${i === 5 && step === 5 ? ` style="opacity:${Math.min(1, progress / 0.35)}"` : ""}>${["qiskit", "pennylane"].includes(art) ? logo(art) : icon(art)}<div><h3>${title}</h3><p>${first}</p><p>${second}</p>${third ? `<p>${third}</p>` : ""}</div></section>`).join("")}</div><div class="feature-resources" data-morph="feature-resources" style="opacity:${references}" aria-hidden="${references === 0}">${qr("core-repo", "github.com/munich-quantum-toolkit/core", "Open Source · MQT and QDMI Communities")}${qr("core", "mqt.readthedocs.io/projects/core", "Tutorials · API · Guides")}</div><aside class="feature-survey" data-morph="feature-survey" style="opacity:${survey};transform:translateY(${12 * (1 - survey)}px)" aria-hidden="${survey === 0}"${survey === 0 ? " inert" : ""}><div class="feature-survey-brand">${logo("unitary")}<span>Quantum Open Source Software</span></div><div><p>Build or Use Quantum Software?</p><h3>Take the QOSS Survey</h3><p class="feature-survey-details">~20 minutes · Open through 29 October 2026</p><a href="https://www.surveymonkey.com/r/QOSSSurvey26">surveymonkey.com/r/QOSSSurvey26</a></div><a class="feature-survey-qr" href="https://www.surveymonkey.com/r/QOSSSurvey26"><img src="${assets["qr-qoss-survey"]}" alt="QR code: 2026 QOSS Survey"/></a></aside>`;
  }
  function separator(title, subtitle, brand) {
    const brandAbove =
      brand === "mqt" ? logo("mqt", "chapter-brand") : esc(subtitle);
    return `<div class="chapter-separator" data-morph="chapter"><p>${brandAbove}</p><h1>${title}</h1><div class="chapter-rule"></div></div>`;
  }
  const slides = [];
  function add(id, section, title, builds, render, notes = "", playbacks = {}) {
    slides.push({ id, section, title, builds, render, notes, playbacks });
  }
  const opening = window.MQSF_OPENING;
  add(
    "title",
    "Munich Quantum Software Forum · October 2026",
    "System Software for Quantum Computing",
    0,
    opening.title,
    "Introduce MQSC first: CTO and co-founder; also Senior Researcher at TUM CDA. The scope is the full path from quantum devices to users. Next: the ecosystem this software must connect.",
  );
  for (const [mode, title, notes] of [
    [
      "problem",
      "Connecting the Quantum Ecosystem",
      "Follow a researcher from a workstation to multiple quantum providers, then cloud compute and HPC. Add the other users last. The growing web makes bespoke integrations hard to maintain. Next: common interfaces and reusable components.",
    ],
    [
      "stack",
      "A Shared Software Stack",
      "Reveal all frontends, then separate programs from runtime requests. Compilation transforms programs; orchestration coordinates resources and execution. The backend interface supplies constraints and carries jobs and results. Next: put MQSC products into these architectural roles.",
    ],
    [
      "detail",
      "Our Approach: From the Metal to the User",
      "Build from QDMI and its devices through the MQT Compiler Collection to MQSC Orchestrator. Both upper components live in MQT Core. Credit the projects' origins and wider communities; finish with the principles. Next: follow a practical hybrid application.",
    ],
  ]) {
    add(
      mode,
      "01 · System Software",
      title,
      opening.phases[mode].length - 1,
      (s, t) => opening.render(mode, s, t ?? 1),
      notes,
      opening.playbacks[mode],
    );
  }
  add(
    "hybrid-chapter",
    "02 · A Practical Hybrid Application",
    "A Classical–Quantum Workflow",
    0,
    () =>
      `<div class="chapter-separator" data-morph="chapter"><p>${appHeading} · QC-AFQMC</p><h1>A Classical–Quantum<br>Workflow</h1><div class="chapter-rule"></div></div>`,
    "Introduce QC-AFQMC as a practical hybrid software workflow. Next: show where classical preparation, quantum sampling, and classical propagation run, and what passes between them.",
  );
  add(
    "hybrid-algorithm",
    "02 · A Practical Hybrid Application",
    appHeading,
    2,
    (s, t) => applicationHeading() + applicationScene(s, t ?? 1),
    "Follow the three panels: prepare the molecular problem and random bases; collect quantum shadows; reuse those observations in classical walker propagation. Credit the paper and Amazon Braket example. H₂/STO-3G makes the software workflow tractable. Next: its actual code and execution.",
    {
      0: { duration: 2200, transition: 900 },
      1: { duration: 3000, transition: 700 },
      2: { duration: 3500, transition: 700 },
    },
  );
  add(
    "afqmc-execution",
    "02 · A Practical Hybrid Application",
    appHeading,
    4,
    (s, t) => applicationHeading() + hybridExecution(s, t ?? 1),
    "Follow preparation, then one QDMI job with 4,096 programs and 4,194,304 shots, then four Slurm ranks propagating 128 walkers per trial. Let both curves finish. These are real jobs using ideal DDSIM and containerized Slurm nodes; timing is rescaled. Next: compile this payload for device constraints.",
    {
      1: { duration: 6500, transition: 0 },
      2: quantumPlayback,
      3: { duration: 12000, transition: 850 },
    },
  );
  add(
    "compile-chapter",
    "03 · Compilation",
    "From Quantum Program to Executable",
    0,
    () =>
      separator(
        "From Quantum Program<br>to Executable",
        "Targeting a Quantum Device",
        "compiler",
      ),
    "A physical target adds connectivity, native operations, and a program format. Keep one submitted AFQMC payload and follow how the compiler adapts it to those constraints.",
  );
  add(
    "routing",
    "03 · Compiling for a Device",
    "From Quantum Program to Executable",
    9,
    (s, t) =>
      heading("From Quantum Program to Executable") +
      compilationScene(s, t ?? 1),
    "Keep input and optimization fixed. Follow Emerald placement, routing, synthesis, and QIR; then switch to IBM/QPY and IonQ/OpenQASM. Local QDMI models use provider snapshots. Compiled QIR is checked on DDSIM; QPY is separately round-trip checked. Next: preserve the structure of larger programs.",
    {
      4: { duration: 6500, transition: 800 },
    },
  );
  add(
    "structure-chapter",
    "04 · Structured Quantum Programs",
    "Why Are We Unrolling?",
    0,
    () =>
      separator(
        "Why Are We Unrolling?",
        "Preserving Structure Through Compilation",
        "jeff",
      ) + paperFooter(),
    "Introduce loops and classical control flow as useful program structure, especially for repeated error-correction cycles and larger algorithms. Credit the paper collaborators. Next: use Grover to show the cost of unrolling.",
  );
  add(
    "grover-growth",
    "04 · Structured Quantum Programs",
    "Why Are We Unrolling?",
    0,
    (s, t) => heading("Why Are We Unrolling?") + growthScene(t ?? 1),
    "Read the Grover loop, then let the unrolled and structure-preserving curves appear. At 22 qubits: about 2.32 MB versus 1.25 KB of emitted OpenQASM. The graph measures program size, not runtime. Next: loops that contain measurements and decisions.",
    { 0: { duration: 6500, transition: 700 } },
  );
  add(
    "structured",
    "04 · Structured Quantum Programs",
    "Structured Programs, All the Way Down",
    4,
    (s) => heading("Structured Programs, All the Way Down") + structureScene(s),
    "Read the syndrome source, reveal its complete QIR entry point, then connect reset, feedback, and loop highlights. The scratch bit stays local to the loop. This portable path does not map or synthesize for a device. Next: execute those control structures and classical calculations.",
  );
  add(
    "execution-chapter",
    "05 · Runtime",
    "From Compilation to Execution",
    0,
    () =>
      separator(
        "From Compilation<br>to Execution",
        "A Runtime for Structured Quantum Programs",
        "runtime",
      ),
    "DDSIM's QIR runtime executes classical control flow together with quantum operations. Introduce the JIT, then use iterative phase estimation to follow the running program.",
  );
  add(
    "adaptive-execution",
    "05 · Adaptive QIR Execution",
    "A Runtime for Structured Quantum Programs",
    1,
    (s, t) =>
      heading("A Runtime for Structured Quantum Programs") +
      adaptiveRuntime(s, t ?? 1),
    "First read the nested loops: four eigenstate qubits plus one recycled control, eight output bits, phase 1/3. Then start the QDMI/JIT replay and follow four workers accumulating 8,192 shots. Calls and histogram share recorded batch completions. Let the distribution settle. Next: the MQT Core capabilities behind these examples.",
    { 1: { duration: 9500, transition: 700 } },
  );
  add(
    "features-chapter",
    "System Software for Quantum Computing",
    "Built for the Quantum Ecosystem",
    0,
    () => separator("Built for the Quantum<br>Ecosystem", "MQT Core", "mqt"),
    "Return from the individual examples to the reusable software behind them. Next: collect the capabilities users and developers can work with today.",
  );
  add(
    "features",
    "System Software for Quantum Computing",
    "Built for the Quantum Ecosystem",
    5,
    (s, t) =>
      heading("Built for the Quantum Ecosystem") + featureOverview(s, t ?? 1),
    "Build through OpenQASM, QIR, Qiskit, PennyLane, installation, and developer resources. F16.6 ends with the QOSS survey invitation: anyone building or using quantum software; about 20 minutes; open through 29 October. Pause for scanning. Next: the community behind the stack.",
    { 5: { duration: 1600 } },
  );
  add(
    "ecosystem",
    "System Software for Quantum Computing",
    "System Software for Quantum Computing",
    0,
    opening.conclusion,
    "Return to the product tower: QDMI, the MQT Compiler Collection, and MQSC Orchestrator. Built with the Community. Driven into practice by MQSC. Credit the people and organizations shown, thank the audience, and leave the closing image visible for discussion.",
  );

  function resize() {
    document.documentElement.style.setProperty(
      "--scale",
      Math.min(innerWidth / 1920, innerHeight / 1080),
    );
  }
  function render(progress = null, transition = 0) {
    state.token++;
    state.animation?.cancel();
    state.animation = null;
    if (progress !== null && !transition) window.MQSF_MOTION.finish();
    draw(progress, progress === null ? 950 : transition);
    const slide = slides[state.slide];
    $("section-label").textContent = slide.section;
    $("deck").classList.toggle("title-slide", state.slide === 0);
    $("slide-number").innerHTML =
      `${String(state.slide + 1)} / ${slides.length}<span class="build-dots" aria-label="Build ${state.step + 1} of ${slide.builds + 1}">${Array.from({ length: slide.builds + 1 }, (_, i) => (i === state.step ? "<b>•</b>" : "•")).join("")}</span>`;
    $("build-progress").style.width =
      `${((state.slide + (state.step + 1) / (slide.builds + 1)) / slides.length) * 100}%`;
    $("speaker-notes").textContent =
      `${state.slide + 1}. ${slide.title} — ${slide.notes}`;
    history.replaceState(null, "", `#${state.slide + 1}.${state.step}`);
  }
  function draw(progress = null, morph = 0) {
    const html = slides[state.slide].render(state.step, progress);
    if (morph) window.MQSF_MOTION.replace($("slide"), html, morph);
    else window.MQSF_MOTION.update($("slide"), html);
    $("slide").dataset.slide = String(state.slide + 1);
    $("slide").dataset.step = String(state.step);
    $("slide").dataset.progress = progress === null ? "" : String(progress);
  }
  function animateTimeline(playback) {
    const token = state.token,
      duration = playback.duration;
    let begin, pending;
    let transitioning = !!playback.transition;
    const cancel = () => cancelAnimationFrame(pending);
    const end = () => {
      if (token !== state.token) return;
      cancel();
      window.MQSF_MOTION.finish();
      state.animation = null;
      draw();
    };
    state.animation = { end, cancel, loop: !!playback.loop };
    function frame(now) {
      if (token !== state.token || !state.animation) return;
      // RAF timestamps share a frame clock; performance.now() may be ahead of it.
      begin ??= now;
      if (transitioning) {
        // Let shared geometry reach the initial frame before replay updates it.
        if (now - begin < playback.transition) {
          pending = requestAnimationFrame(frame);
          return;
        }
        window.MQSF_MOTION.finish();
        transitioning = false;
        begin = now;
      }
      const elapsed = Math.max(0, now - begin),
        cycle = playback.loop ? elapsed % (duration + 2400) : elapsed;
      draw(Math.min(1, cycle / duration));
      if (playback.loop || elapsed < duration)
        pending = requestAnimationFrame(frame);
      else end();
    }
    pending = requestAnimationFrame(frame);
  }
  function play(playback) {
    if (!playback || matchMedia("(prefers-reduced-motion: reduce)").matches) {
      render();
      return;
    }
    render(0, playback.transition || 0);
    animateTimeline(playback);
  }
  function go(slide, step = 0, forward = false) {
    state.slide = Math.max(0, Math.min(slide, slides.length - 1));
    state.step = Math.max(0, Math.min(step, slides[state.slide].builds));
    play(forward ? slides[state.slide].playbacks[state.step] : null);
  }
  function next() {
    if (state.animation) {
      state.animation.end();
    }
    const current = slides[state.slide];
    if (state.step < current.builds) {
      state.step++;
      play(current.playbacks[state.step]);
    } else if (state.slide < slides.length - 1) go(state.slide + 1, 0, true);
  }
  function previous() {
    if (state.step > 0) go(state.slide, state.step - 1);
    else if (state.slide > 0)
      go(state.slide - 1, slides[state.slide - 1].builds);
  }
  function overview(open) {
    const d = $("overview");
    if (!open) {
      d.close();
      $("deck").focus();
      return;
    }
    state.animation?.end();
    state.overview = state.slide;
    $("slide-grid").innerHTML = slides
      .map(
        (s, i) =>
          `<button data-slide="${i}" class="${i === state.overview ? "selected" : ""}"><b>${i + 1}</b>${esc(s.title)}</button>`,
      )
      .join("");
    $("slide-grid")
      .querySelectorAll("button")
      .forEach((b) =>
        b.addEventListener("click", () => {
          go(Number(b.dataset.slide));
          overview(false);
        }),
      );
    d.showModal();
    $("slide-grid").children[state.overview].focus();
  }
  document.addEventListener("keydown", (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    if ($("overview").open) {
      if (e.key === "Escape") {
        e.preventDefault();
        overview(false);
        return;
      }
      const offset = {
        ArrowLeft: -1,
        ArrowRight: 1,
        ArrowUp: -5,
        ArrowDown: 5,
      }[e.key];
      if (offset) {
        e.preventDefault();
        state.overview = Math.max(
          0,
          Math.min(slides.length - 1, state.overview + offset),
        );
        $("slide-grid")
          .querySelectorAll("button")
          .forEach((b, i) =>
            b.classList.toggle("selected", i === state.overview),
          );
        $("slide-grid").children[state.overview].focus();
      }
      if (e.key === "Enter") {
        e.preventDefault();
        go(state.overview);
        overview(false);
      }
      return;
    }
    if (/^[0-9]$/.test(e.key)) {
      state.jump = (state.jump + e.key).slice(-2);
      $("jump-indicator").textContent = `Go to ${state.jump} ↵`;
      $("jump-indicator").hidden = false;
      return;
    }
    if (e.key === "Enter" && state.jump) {
      e.preventDefault();
      go(Number(state.jump) - 1);
      state.jump = "";
      $("jump-indicator").hidden = true;
      return;
    }
    if (["ArrowRight", "PageDown", " ", "Enter"].includes(e.key)) {
      e.preventDefault();
      next();
    }
    if (["ArrowLeft", "PageUp", "Backspace"].includes(e.key)) {
      e.preventDefault();
      previous();
    }
    if (e.key === "Home") {
      e.preventDefault();
      go(0);
    }
    if (e.key === "End") {
      e.preventDefault();
      go(slides.length - 1, slides.at(-1).builds);
    }
    if (e.key === "Escape" || e.key.toLowerCase() === "g") {
      e.preventDefault();
      state.jump = "";
      $("jump-indicator").hidden = true;
      overview(true);
    }
    if (e.key.toLowerCase() === "b")
      $("blackout").hidden = !$("blackout").hidden;
    if (e.key.toLowerCase() === "p")
      $("speaker-notes").hidden = !$("speaker-notes").hidden;
    if (e.key.toLowerCase() === "f")
      document.fullscreenElement
        ? document.exitFullscreen()
        : document.documentElement.requestFullscreen();
  });
  let touchX = 0;
  document.addEventListener(
    "touchstart",
    (e) => {
      touchX = e.changedTouches[0].screenX;
    },
    { passive: true },
  );
  document.addEventListener(
    "touchend",
    (e) => {
      const dx = e.changedTouches[0].screenX - touchX;
      if (Math.abs(dx) > 80) dx < 0 ? next() : previous();
    },
    { passive: true },
  );
  addEventListener("resize", resize);
  addEventListener("beforeprint", () => {
    state.animation?.end();
    $("print-deck").innerHTML = slides
      .map((s, i) => {
        const footer = document
          .querySelector("#deck > .slide-footer")
          .cloneNode(true);
        const number = footer.querySelector("#slide-number");
        number.removeAttribute("id");
        number.className = "print-slide-number";
        number.textContent = `${i + 1} / ${slides.length}`;
        return `<section class="print-slide"><header class="slide-header"><span>${esc(s.section)}</span>${logo("mqsc")}</header><article data-slide="${i + 1}" data-step="${s.builds}">${s.render(s.builds)}</article>${footer.outerHTML}</section>`;
      })
      .join("");
  });
  addEventListener("afterprint", () => {
    $("print-deck").replaceChildren();
  });
  window.MQSF_DECK = {
    slides: slides.map((s) => ({
      id: s.id,
      title: s.title,
      playbacks: s.playbacks,
      builds: s.builds,
      notes: s.notes,
    })),
    getState: () => ({
      slide: state.slide,
      step: state.step,
      animating: !!state.animation,
    }),
    go,
    next,
    previous,
    finishAnimation: () => state.animation?.end(),
    finishMotion: () => window.MQSF_MOTION.finish(),
  };
  $("mqsc-logo").src = assets.mqsc;
  resize();
  const hash = location.hash.match(/^#(\d+)(?:\.(\d+))?$/);
  go(hash ? Number(hash[1]) - 1 : 0, hash ? Number(hash[2] || 0) : 0);
  $("deck").focus();
})();
