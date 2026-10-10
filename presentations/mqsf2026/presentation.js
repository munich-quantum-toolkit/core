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
  const artifact = (id, unrolled = false) =>
    [
      ...(variant("parity", unrolled).stages || []),
      ...(variant("parity", unrolled).exports || []),
    ].find((a) => a.id === id);
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
  const metric = (value, label, wide = false) =>
    `<div class="metric"><div class="value ${wide ? "long" : ""}">${esc(value)}</div><div class="label">${esc(label)}</div></div>`;
  const qr = (name, url, caption) =>
    `<div class="qr-block"><img src="${assets[`qr-${name}`]}" alt="QR code: ${esc(url)}"/><div><div class="url">${esc(url)}</div><div class="caption">${esc(caption)}</div></div></div>`;
  const fmt = (n, digits = 2) =>
    Number(n).toLocaleString("en-US", { maximumFractionDigits: digits });
  const sourceNote = (s) => `<p class="source-line">${s}</p>`;
  function code(
    a,
    { start = 0, count = 11, hot = [], title, compact = false } = {},
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
    return `<div data-morph="code-panel" class="code-frame ${compact ? "compact" : ""}"><div class="code-title">${esc(title || a.label || a.language || "Actual source")}</div><pre class="highlight">${body}</pre></div>`;
  }
  const focus = (a, needle, before = 1) =>
    Math.max(
      0,
      (a?.code?.split("\n").findIndex((s) => s.includes(needle)) ?? 0) - before,
    );
  const flatten = window.MQSF_CIRCUIT.flatten;
  const circuit = window.MQSF_CIRCUIT.render;
  function histogram(capture, elapsed = Infinity, box = "0 0 900 480") {
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
    const x0 = 65,
      w = 790,
      plotH = 285,
      y0 = 360,
      bw = w / values.length;
    let body = line(x0, y0, x0 + w, y0, "#9cb3c9", 2);
    [0, 0.5, 1].forEach((f) => {
      const y = y0 - f * plotH;
      body +=
        line(x0, y, x0 + w, y, "#e0e9f2", 1) +
        text(x0 - 12, y + 8, String(Math.round(max * f)), 23, "#5b7088", "end");
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
      if (n > max * 0.07) body += text(x + (bw - 9) / 2, y0 - h - 10, n, 22);
      if (i === 0 || i === values.length - 1 || i % 2 === 0 || key === "85")
        body += text(x + (bw - 9) / 2, y0 + 35, key, 22, "#5b7088");
    });
    body +=
      text(450, 428, "8-bit phase estimate · integer / 256", 27, "#5b7088") +
      text(
        450,
        475,
        `${fmt(shots.length, 0)} / ${fmt(capture.num_shots || 0, 0)} actual shots`,
        31,
        "#2f70b8",
      );
    return `<div class="histogram" data-shots="${shots.length}" data-observed-bins="${all.length}">${svg(body, box, "Actual QPE measurement distribution")}</div>`;
  }
  function sequence(capture, elapsed = Infinity) {
    const events = capture.events || [];
    const groups =
      capture.trace_kind === "python-observed-adapter-calls"
        ? [
            ["Submit program collection", "try_submit_job"],
            ["Wait and collect", "PennyLaneJob.result"],
            ["Read indexed results", "indexed shots"],
          ]
        : [
            ["Create session", "session_init"],
            ["Create job", "create_device_job"],
            ["Attach program", "set_programs"],
            ["Submit", "job_submit"],
            ["Wait / execute", "job_wait"],
            ["Read results", "get_results"],
          ];
    let body =
      text(
        130,
        28,
        capture.trace_kind === "python-observed-adapter-calls"
          ? "PennyLane / Core"
          : "Client",
        27,
      ) +
      text(650, 28, "DDSIM", 27) +
      line(130, 50, 130, 340, "#b8ccdf", 2) +
      line(650, 50, 650, 340, "#b8ccdf", 2);
    groups.forEach(([label, key], i) => {
      const e = events.find((e) => e.operation.includes(key)),
        done = e && elapsed >= e.time_ms,
        inFlight =
          e && elapsed >= e.time_ms && elapsed < e.time_ms + e.duration_ms;
      const y = 76 + i * (groups.length === 3 ? 90 : 48);
      body += `<g opacity="${done ? 1 : 0.18}">${line(130, y, 650, y, inFlight ? "#078b9c" : "#2f70b8", inFlight ? 5 : 3)}<path d="M640 ${y - 7}L651 ${y}L640 ${y + 7}" fill="none" stroke="#2f70b8" stroke-width="3"/>${text(390, y - 10, label, 25)}</g>`;
    });
    return `<div class="sequence-svg">${svg(body, "0 0 780 350", "Real QDMI call sequence")}</div>`;
  }
  function runtimeCode(capture, elapsed = Infinity) {
    const e = [...(capture.events || [])]
      .reverse()
      .find((e) => e.time_ms <= elapsed);
    const a = {
      code: capture.client_source || "",
      lines_html: capture.client_source_lines_html,
    };
    const python = capture.trace_kind === "python-observed-adapter-calls";
    const srcLine = (e?.source_line || 1) - 1;
    return code(a, {
      start: python ? 0 : Math.max(0, srcLine - 3),
      count: python ? 8 : 9,
      hot: [srcLine],
      compact: true,
      title: python
        ? "Actual Core adapter · native batch submission"
        : "Actual client · QDMI C ABI",
    });
  }
  const totalTime = () =>
    Math.max(
      run.completed_ms || 0,
      ...(run.events || []).map((e) => e.time_ms + e.duration_ms),
    );
  function executionScene(elapsed, mode) {
    const finish = elapsed >= totalTime(),
      ratio = Math.min(1, elapsed / totalTime());
    return `<div class="execution-grid"><div>${runtimeCode(run, elapsed)}${sequence(run, elapsed)}</div><div><div class="runtime-strip"><span>${esc(mode)}</span><span>${fmt(Math.min(elapsed, totalTime()), 1)} / ${fmt(totalTime(), 1)} ms</span></div><div class="clock-line"><div style="width:${ratio * 100}%"></div></div>${histogram(run, elapsed)}<p class="result-summary">${finish ? "The same payload. A complete distribution." : "Each bar changes at a recorded shot completion."}</p></div></div>${sourceNote("DDSIM · instrumented serial capture · one clock for API calls and actual shot completion · no simulated timings")}`;
  }
  function energyChart(step = 3) {
    const p = app.propagation;
    if (!p) return "";
    const curves = p.curves,
      ref = app.chemistry.fci_energy;
    // Full recorded range, including the early finite-walker outliers.
    const values = curves
      .flatMap((c, j) =>
        c.energy.flatMap((e, i) => [
          e - ref - (j ? 0 : c.stderr[i]),
          e - ref + (j ? 0 : c.stderr[i]),
        ]),
      )
      .map((e) => e * 1000);
    const min = Math.floor(Math.min(...values) / 25) * 25,
      max = Math.ceil(Math.max(...values) / 25) * 25;
    const x0 = 105,
      y0 = 440,
      w = 1160,
      h = 345,
      tmax = p.tau.at(-1),
      x = (t) => x0 + (t / tmax) * w,
      y = (e) => y0 - ((e - min) / (max - min)) * h;
    let body =
      line(x0, y0, x0 + w, y0, "#8daac5", 2) +
      line(x0, y0, x0, y0 - h, "#8daac5", 2);
    for (let e = min; e <= max; e += 25)
      body +=
        line(x0, y(e), x0 + w, y(e), "#e1eaf2", 1) +
        text(x0 - 20, y(e) + 8, e, 26, "#5b7088", "end");
    Array.from({ length: 5 }, (_, i) => (i / 4) * tmax).forEach((t) => {
      body += text(x(t), y0 + 39, fmt(t, 1), 28, "#5b7088");
    });
    body +=
      `<path class="reference" d="M${x0} ${y(0)}H${x0 + w}"/>` +
      text(1280, y(0) + 8, "FCI", 27, "#078b9c", "start") +
      text(675, 525, "Imaginary time τ (Ha⁻¹)", 30) +
      text(105, 43, "Energy − FCI (mHa)", 30, "#142b45", "start");
    const count = Math.max(
      2,
      Math.min(p.tau.length, Math.ceil((step / 3) * p.tau.length)),
    );
    curves.forEach((c, j) => {
      const points = c.energy
        .slice(0, count)
        .map((e, i) => `${x(p.tau[i])},${y((e - ref) * 1000)}`);
      if (!j) {
        const upper = c.energy
            .slice(0, count)
            .map(
              (e, i) => `${x(p.tau[i])},${y((e + c.stderr[i] - ref) * 1000)}`,
            ),
          lower = c.energy
            .slice(0, count)
            .map(
              (e, i) => `${x(p.tau[i])},${y((e - c.stderr[i] - ref) * 1000)}`,
            )
            .reverse();
        body += `<polygon class="error" points="${[...upper, ...lower].join(" ")}"/>`;
      }
      body += `<polyline points="${points.join(" ")}" stroke="${j ? "#829bb3" : "#2f70b8"}" stroke-width="${j ? 3 : 5}" fill="none" ${j ? 'stroke-dasharray="9 6"' : ""}/>`;
    });
    body +=
      line(760, 35, 815, 35) +
      text(830, 44, "Shadow trial", 27, "#142b45", "start");
    body +=
      line(1075, 35, 1130, 35, "#829bb3", 3) +
      text(1145, 44, "HF trial", 27, "#5b7088", "start") +
      text(1350, 270, "Final snapshot", 25, "#5b7088", "start") +
      text(
        1350,
        310,
        fmt((curves[0].energy.at(-1) - ref) * 1000, 2) + " mHa",
        32,
        "#2f70b8",
        "start",
      ) +
      text(
        1350,
        350,
        "± " + fmt(curves[0].stderr.at(-1) * 1000, 2) + " mHa",
        28,
        "#5b7088",
        "start",
      );
    return `<div class="energy-chart">${svg(body, "0 0 1640 555", "All recorded AFQMC energy estimates and shadow-trial walker-only standard error")}</div>`;
  }
  const appCircuit = () =>
    app.scenario?.variants?.[0]?.stages?.find((a) => a.circuit)?.circuit;
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
  function walkerScene(progress = 1) {
    const p = app.propagation,
      frames = p?.curves?.[0]?.frames || [];
    if (!frames.length) return "";
    const index = Math.max(
        0,
        Math.min(frames.length - 1, progress * (frames.length - 1)),
      ),
      low = Math.floor(index),
      high = Math.min(low + 1, frames.length - 1),
      t = index - low;
    const f = frames[low],
      next = frames[high],
      tau = f.tau + (next.tau - f.tau) * t;
    const weights = f.walkers.map(
        (w, i) => w.weight + (next.walkers[i].weight - w.weight) * t,
      ),
      max = Math.max(...weights);
    const count = f.walkers.length,
      columns = 16,
      rows = Math.ceil(count / columns),
      dx = 67,
      dy = Math.min(67, 430 / rows);
    let body = "";
    body +=
      `<rect x="35" y="45" width="1120" height="${rows * dy + 80}" rx="18" fill="#f0f6fb"/>` +
      text(600, 82, "Walker ensemble · schematic arrangement", 25, "#526d88");
    f.walkers.forEach((w, i) => {
      const worker = Math.floor((i % columns) / 4),
        xx = 69 + (i % columns) * 67 + worker * 17,
        yy = 125 + Math.floor(i / columns) * dy;
      const r = 27 * Math.sqrt(weights[i] / max);
      body += `<g data-morph="walker-${w.id}"><circle cx="${xx}" cy="${yy}" r="${r}" fill="#2f70b8" opacity="${0.25 + (0.75 * weights[i]) / max}"/><circle cx="${xx}" cy="${yy}" r="2.3" fill="white"/></g>`;
    });
    body += text(
      600,
      rows * dy + 157,
      `τ = ${fmt(tau, 2)} Ha⁻¹     ·     step ${Math.round(progress * (p.steps - 1))} / ${p.steps - 1}`,
      32,
      "#2f70b8",
    );
    return svg(
      body,
      `0 0 1200 ${rows * dy + 190}`,
      "Measured AFQMC walker weights replayed over imaginary time",
    );
  }
  function batchScene(step, progress = null) {
    const capture = app.execution || {},
      events = capture.events || [],
      duration = capture.duration_ms || 1;
    const submit = events.find((e) => e.operation.includes("try_submit_job")),
      begin = submit?.time_ms ?? 0,
      elapsed = begin + (progress ?? 1) * (duration - begin);
    const preparing = elapsed < (submit?.time_ms ?? 0);
    const client = preparing
      ? code(
          {
            code: app.batch_source,
            lines_html: app.batch_source_lines_html,
            label: "Actual PennyLane broadcast call",
          },
          { count: 3, compact: true },
        )
      : runtimeCode(capture, elapsed);
    const retrieved = events.filter(
      (e) =>
        Number.isInteger(e.program_index) &&
        elapsed >= e.time_ms + e.duration_ms,
    );
    const count = retrieved.length,
      snapshots = app.snapshots || [],
      cells = 128,
      perCell = Math.ceil(snapshots.length / cells);
    let body = "";
    for (let i = 0; i < cells; i++) {
      const x = 20 + (i % 16) * 44,
        y = 24 + Math.floor(i / 16) * 38;
      body += `<rect x="${x}" y="${y}" width="31" height="25" rx="4" fill="${i * perCell < count ? "#2f70b8" : "#d7e5f2"}"/>`;
    }
    body += text(
      360,
      357,
      `${fmt(count, 0)} / ${fmt(snapshots.length, 0)} program results`,
      30,
      "#2f70b8",
    );
    body += text(
      360,
      391,
      `Each tile groups ${perCell} indexed programs`,
      22,
      "#526d88",
    );
    return `<div class="execution-grid batch-grid" data-programs="${count}"><div>${client}<p class="batch-phase">${preparing ? "Prepare and lower the circuit collection" : count === snapshots.length ? "All indexed samples retrieved" : "Submit → wait → retrieve by program index"}</p>${sequence(capture, elapsed)}</div><div><div class="runtime-strip"><span>QDMI window · slowed replay</span><span>${fmt(elapsed - begin, 1)} / ${fmt(duration - begin, 1)} ms</span></div><div class="clock-line"><div style="width:${((elapsed - begin) / (duration - begin)) * 100}%"></div></div><div class="batch-result">${svg(body, "0 0 750 420", "Captured indexed program results arriving at their actual retrieval times")}</div><div class="metric-row">${metric(fmt(app.workload?.snapshots, 0), "programs")}${metric(fmt(count * (app.workload?.shots_per_snapshot || 0), 0), "shots retrieved")}</div></div></div>`;
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
      full ? "All-to-all connectivity" : "Device connectivity · physical sites",
      24,
      "#526d88",
    );
    const r = sites.length > 80 ? 12 : sites.length > 40 ? 17 : 20;
    body += sites
      .map((s) => {
        const p = points.get(s.id),
          hot = active.has(s.id);
        return `<g data-morph="device-node-${s.id}" transform="translate(${p.x} ${p.y})"><circle r="${r}" fill="${hot ? "#d4e9fa" : "#ecf3fa"}" stroke="${hot ? "#087d92" : "#92b2d0"}" stroke-width="${hot ? 4 : 1.5}"/>${text(0, 6, s.id, r > 15 ? 17 : 12, "#58718b")}</g>`;
      })
      .join("");
    (layout || []).slice(0, 6).forEach((site, i) => {
      const destination = points.get(site),
        origin = points.get(blend?.before?.[i]) || destination;
      if (!destination) return;
      const t = blend?.fraction ?? 1,
        px = origin.x + (destination.x - origin.x) * t,
        py = origin.y + (destination.y - origin.y) * t;
      body += `<g data-morph="logical-qubit-${i}" transform="translate(${px} ${py})"><circle r="${r + 5}" fill="${colors[i]}" stroke="white" stroke-width="3"/>${text(0, 7, `q${i}`, 17, "white")}</g>`;
    });
    return `<div class="target-map" data-morph="target-map">${svg(body, "0 0 820 630", `${target?.label || "Emerald"} captured coupling graph`)}</div>`;
  }
  function targetSummary(target) {
    const m = target?.metadata || {},
      names = (
        target?.id === "aws.ionq.forte-1"
          ? target.compiler_model.operations.filter((o) => o.name !== "measure")
          : m.operations || []
      ).map((o) => (typeof o === "string" ? o : o.name));
    return `<div class="device-properties"><div><b>${m.qubits || m.sites?.length || 0}</b><span>qubits</span></div><div><b>${m.edges?.length || 0}</b><span>couplings</span></div><div class="gate-set"><b>${esc(names.join(" · "))}</b><span>native operations</span></div></div>`;
  }
  function routingScene(step, progress = null, target = targets[0]) {
    const comp = target?.compilation || {},
      events = comp.routing_trace || [];
    const search = events.filter((e) =>
      ["forward", "backward", "score", "selected"].includes(e.phase),
    );
    const cursor = (progress ?? 1) * Math.max(1, search.length),
      index = Math.min(search.length - 1, Math.floor(cursor));
    const event = search[index] || {},
      initial = (comp.layout?.initial || variant().layout?.initial || []).slice(
        0,
        6,
      );
    const fraction =
      progress === null
        ? 1
        : 1 - Math.pow(1 - (cursor - Math.floor(cursor)), 3);
    let placement =
      step === 1 ? (event.after || initial).slice(0, 6) : [...initial];
    const stage = targetStage(
      target,
      step >= 3 ? "target-native-synthesis" : "place-and-route",
    );
    const circuitData =
      step < 2
        ? targetStage(target, "optimized")?.circuit || appCircuit()
        : stage?.circuit;
    const ops = window.MQSF_CIRCUIT.schedule(circuitData);
    const lastColumn = Math.max(0, ...ops.map((o) => o.column));
    const layerPosition = Math.min(
      lastColumn,
      (progress ?? 1) * (lastColumn + 1),
    );
    const column = Math.floor(layerPosition),
      selected = step < 2 ? [] : ops.filter((o) => o.column === column);
    const opIndices = selected.map((o) => o.index);
    const physical = (q) => circuitData.qubits.find((w) => w.id === q)?.site;
    let swaps = 0,
      beforeSwap = null;
    if (step === 2) {
      // Independent operations may share a layer; physical-wire dependencies preserve SWAP order.
      for (const o of ops
        .filter((o) => o.column <= column)
        .sort((a, b) => a.column - b.column)) {
        if (o.name !== "swap") continue;
        if (o.column === column && !beforeSwap) beforeSwap = [...placement];
        const [a, b] = o.qubits.map(physical);
        placement = placement.map((site) =>
          site === a ? b : site === b ? a : site,
        );
        swaps++;
      }
    }
    const active =
      step === 0
        ? placement
        : selected
            .flatMap((o) => o.qubits.map(physical))
            .filter(Number.isInteger);
    const activeEdges = new Set(
      selected
        .filter((o) => o.qubits.length === 2)
        .map((o) =>
          o.qubits
            .map(physical)
            .sort((a, b) => a - b)
            .join(":"),
        ),
    );
    const blend =
      step === 1
        ? { before: event.before || search[index - 1]?.after, fraction }
        : beforeSwap
          ? {
              before: beforeSwap,
              fraction:
                progress === null || progress === 1
                  ? 1
                  : layerPosition - column,
            }
          : null;
    const counts = comp.metrics?.["target-native-synthesis"]?.counts || {};
    const nativeSummary = Object.entries(counts)
      .filter(([name]) => name !== "measure")
      .map(([name, count]) => `${count} ${name.toUpperCase()}`)
      .join(" · ");
    const phase =
      step === 1
        ? `${{ forward: "Forward →", backward: "← Backward", score: "Evaluate", selected: "Select" }[event.phase] || event.phase} · trial ${(event.trial ?? 0) + 1}`
        : step === 3
          ? "Native synthesis"
          : step === 2
            ? "Routed operations"
            : "Initial placement";
    return `<div class="architecture-grid"><div><div class="phase-labels">${["Placement", "Forward ↔ backward", "Routing", "Native synthesis"].map((l, i) => `<span class="${i === step ? "selected" : ""}">${l}</span>`).join("")}</div>${circuit(circuitData, { active: opIndices, camera: step >= 2 ? layerPosition : null, limit: 18, key: step < 2 ? "routing-logical" : step === 2 ? "routing-routed" : "routing-native" })}<div class="mapping-legend">${step === 3 ? "Physical wires shown in the circuit" : placement.map((p, i) => `<span>q${i} → ${p}</span>`).join("")}</div><div class="telemetry"><b>${phase}</b><span>${step === 1 ? `Search event ${index + 1} / ${search.length}` : step === 3 ? nativeSummary : "Actual compiler output"}</span>${step < 3 ? `<strong>${step === 0 ? 0 : step === 1 ? (event.swaps ?? events.find((e) => e.phase === "score" && e.trial === event.trial)?.swaps ?? 0) : step === 2 && progress !== null ? swaps : (comp.layout?.swaps?.length ?? 0)}<small> SWAPs</small></strong>` : ""}</div>${programMetrics(target, step < 2 ? "optimized" : step === 2 ? "place-and-route" : "target-native-synthesis")}</div><div>${deviceView(target, active, step === 3 ? null : placement, blend, step >= 1 ? activeEdges : null)}</div></div>`;
  }
  function hpcScene(progress = 1) {
    const p = app.propagation || {},
      tasks = p.tasks || [],
      pids = p.worker_pids || [],
      first = pids.map((pid) =>
        Math.min(
          ...tasks.filter((t) => t.worker_pid === pid).map((t) => t.started_ms),
        ),
      );
    const start = Math.max(...first) - 40,
      end = Math.max(...tasks.map((t) => t.finished_ms)),
      now = start + progress * (end - start);
    const x = (ms) => 180 + ((ms - start) / (end - start)) * 1310;
    let body = "";
    pids.forEach((pid, row) => {
      const yy = 50 + row * 66;
      body +=
        `<rect x="177" y="${yy}" width="1315" height="42" rx="6" fill="#edf3fa"/>` +
        text(157, yy + 28, `Process ${row + 1}`, 23, "#526d88", "end");
      tasks
        .filter(
          (t) =>
            t.worker_pid === pid &&
            t.finished_ms >= start &&
            t.started_ms <= now,
        )
        .forEach((task) => {
          const left = x(Math.max(start, task.started_ms)),
            right = x(Math.min(now, task.finished_ms));
          body += `<rect x="${left}" y="${yy + 3}" width="${Math.max(0, right - left)}" height="36" fill="${task.label === "quantum" ? "#2f70b8" : "#6c91a2"}"><title>PID ${pid}, ${task.label} walker ${task.walker_id}; ${fmt(task.started_ms, 2)}–${fmt(task.finished_ms, 2)} ms</title></rect>`;
        });
    });
    for (let i = 0; i <= 4; i++)
      body += text(
        x(start + (i / 4) * (end - start)),
        355,
        fmt((start + (i / 4) * (end - start)) / 1000, 2) + " s",
        23,
        "#526d88",
      );
    body += line(x(now), 37, x(now), 317, "#087d92", 3);
    const src = {
      code: app.classical_source || "",
      lines_html: app.classical_source_lines_html,
    };
    return `<div class="hpc-lanes">${svg(body, "0 0 1550 385", "Actual four-process AFQMC task intervals on a measured wall clock")}</div><div class="hpc-details"><div>${code(src, { start: focus(src, "for label in", 0), count: 3, compact: true, title: "Actual walker-task submission loop" })}</div><div><div class="metric-row">${metric(p.processes || 0, "CPU processes")}${metric(tasks.length, "walker tasks")}</div><p class="note">Each bar: one complete ${p.steps}-step walker.<br>Blue: shadow trial · grey: Hartree–Fock trial.<br>Measured steady-phase zoom; whole pool: ${fmt((p.duration_ms || 0) / 1000, 2)} s, including startup.</p></div></div>`;
  }
  function programMetrics(target, id) {
    const m = target?.compilation?.metrics?.[id] || {},
      c = targetStage(target, id)?.circuit,
      ops = c ? flatten(c) : [];
    const two =
      m.two_qubit_operations ??
      ops.filter((o) => o.qubits?.length === 2).length;
    return `<div class="program-metrics" data-stage="${id}">${[
      [m.operations ?? ops.length, "operations"],
      [two, "two-qubit"],
      [m.depth ?? 0, "depth"],
      [m.active_qubits ?? c?.qubits?.length ?? 0, "active qubits"],
    ]
      .map(
        ([v, l]) =>
          `<div data-morph="metric-${l.replaceAll(" ", "-")}"><b>${fmt(v, 0)}</b><span>${l}</span></div>`,
      )
      .join("")}</div>`;
  }
  function stageFlow(labels, active, progress = 1) {
    return `<div class="stage-flow" style="--stage:${active};--stages:${labels.length}">${labels.map((l, i) => `<div data-morph="pipeline-${i}" class="${i === active ? "selected" : i < active ? "complete" : ""}"><span>${i + 1}</span>${l}</div>`).join("")}<i style="width:${(100 * (active + progress)) / labels.length}%"></i></div>`;
  }
  // Display folds retain the original line numbers and every control boundary.
  // This changes the view only; full unmodified programs remain in evidence.json.gz.
  function structureCode(a, title, qir = false) {
    const lines = (a?.code || "").split("\n"),
      rows = [];
    const start = qir ? lines.findIndex((l) => l.startsWith("define ")) : 0;
    let end = qir
      ? lines.findIndex((l, i) => i > start && l === "}") + 1
      : lines.length;
    if (end <= start) end = lines.length;
    const foldable = (l) =>
      qir
        ? l.trim() &&
          !/^define|^\}|^\w+:/u.test(l) &&
          !/(?:\b(?:br |phi |icmp |ret |add )|__(?:reset|mz)__|__read_result)/u.test(
            l,
          )
        : /^\s*(?:r\(|ctrl @ z |rx\(|rz\()/u.test(l);
    for (let i = Math.max(0, start); i < end;) {
      if (!lines[i].trim()) {
        i++;
        continue;
      }
      let last = i + 1;
      if (foldable(lines[i]))
        while (last < end && foldable(lines[last])) last++;
      if (last - i > 1)
        rows.push(
          `<span class="code-line folded"><span class="line-no">${i + 1}–${last}</span><span class="code-text">${" ".repeat(lines[i].match(/^ */)[0].length)}⋯ ${last - i} ${qir ? "straight-line instructions" : "gate operations"}</span></span>`,
        );
      else
        rows.push(
          `<span class="code-line ${/for |if |while |br |phi |icmp |measure|reset|read_result/.test(lines[i]) ? "hot" : ""}"><span class="line-no">${i + 1}</span><span class="code-text">${qir && /^\w+:/.test(lines[i]) ? esc(lines[i].replace(/ {2,};/, "  ;")) : (a.lines_html?.[i] ?? esc(lines[i]))}</span></span>`,
        );
      i = last;
    }
    const middle = Math.ceil(rows.length / 2);
    return `<div class="code-frame structure-code ${qir ? "qir-structure" : ""}" data-lines="${rows.length}"><div class="code-title">${title}</div><div class="${qir ? "code-columns" : ""}"><pre class="highlight">${(qir ? rows.slice(0, middle) : rows).join("")}</pre>${qir ? `<pre class="highlight">${rows.slice(middle).join("")}</pre>` : ""}</div><p class="code-foot">${qir ? "Complete entry-point control flow · label spacing condensed" : "Complete program structure"} · explicit folds · original line numbers</p></div>`;
  }
  function afqmcExecution(s, t) {
    const progress = t ?? 1;
    return `<div class="hybrid-execution ${s ? "cpu-focus" : ""}"><div class="quantum-pane"><div class="location-label">Quantum device · DDSIM capture</div>${batchScene(1, s ? 1 : progress)}</div><aside class="cpu-pane"><div class="location-label">Classical CPUs</div>${walkerScene(s ? progress : 0)}<div class="cpu-algorithm"><b>Parallel walker propagation</b><p>Propagate → trial overlap<br>→ reweight → reduce energy</p></div>${metric(app.propagation?.processes || 0, "local worker processes")}<p class="note">Measured quantum data is reused at every imaginary-time step.</p></aside></div>${s ? `<div class="integrated-hpc">${hpcScene(progress)}</div>` : ""}`;
  }
  function targetSwitchScene(s, t) {
    const target = targets[Math.min(s, targets.length - 1)] || {},
      progress = t ?? 1;
    const phase =
      progress < 0.2 ? 0 : progress < 0.45 ? 1 : progress < 0.72 ? 2 : 3;
    const ids = [
      "optimized",
      "optimized",
      "place-and-route",
      "target-native-synthesis",
    ];
    const a = targetStage(target, ids[phase]),
      circuitData = a?.circuit;
    const label = [
      "Query target",
      "Place qubits",
      "Route operations",
      "Synthesize gates",
    ];
    return `<div class="device-id" data-morph="device-id">device_id = <b>"${esc(target.id)}"</b></div>${stageFlow(label, phase)}<div class="architecture-grid device-recompile"><div><h3>${esc(target.label)}</h3>${targetSummary(target)}${circuit(circuitData, { limit: 17, key: `switch-${phase}`, label: label[phase] })}${programMetrics(target, ids[phase])}<p class="note">${phase === 3 ? "Compiled for this gate set and topology." : "The same input program is recompiled for the selected target."}</p></div><div>${deviceView(target, phase ? target.compilation?.layout?.initial || [] : [], phase ? target.compilation?.layout?.initial : null)}<div class="device-links">${qr(s === 0 ? "iqm" : s === 1 ? "ibm" : "braket", s === 0 ? "iqm-finland.github.io/QDMI-on-IQM" : s === 1 ? "ibm-qdmi-device.readthedocs.io" : "amazon-braket-qdmi-device.readthedocs.io", "Device integration through QDMI")}</div></div></div>`;
  }
  const slides = [];
  function add(id, section, title, builds, render, notes = "", playbacks = {}) {
    slides.push({ id, section, title, builds, render, notes, playbacks });
  }
  add(
    "title",
    "Munich Quantum Software Forum · October 2026",
    "System Software for Quantum Computing",
    0,
    () =>
      `<div class="title-layout"><h1>System Software for Quantum Computing:<br><span class="blue">From the Metal to the User</span></h1><div class="title-byline"><b>Lukas Burgholzer</b><span>CTO &amp; Co-founder · MQSC</span><span>Technical University of Munich</span></div><div class="title-animation" data-morph="world">${viz.architecture("title")}</div></div>`,
    "Introduce MQSC: system software connects users, classical computing infrastructure and heterogeneous quantum systems.",
  );
  add(
    "integration",
    "01 · System software",
    "Hardware scales. Software breaks.",
    2,
    (s) =>
      heading(
        [
          'Hardware scales.<span class="blue"> Software breaks.</span>',
          "A shared software stack",
          "From the metal to the user",
        ][s],
      ) +
      `<div class="scene world-scene" data-morph="world">${viz.architecture(["problem", "stack", "detail"][s])}</div><div class="reference-strip">${qr("mqss", "doi.org/10.1145/3773656.3773669", "Munich Quantum Software Stack · published architecture")}</div>`,
    "Build the common stack in the same world. Users and devices remain in place while bespoke connections become shared interfaces. The second build reveals resource management, compiler infrastructure and QDMI. Resource orchestration belongs to the surrounding system, not to Core alone.",
  );
  add(
    "hybrid-algorithm",
    "02 · A practical hybrid application",
    "Quantum-assisted auxiliary-field Monte Carlo",
    1,
    (s, t = null) =>
      heading(
        "Quantum-assisted auxiliary-field Monte Carlo",
        "A molecular energy calculation connects quantum sampling and parallel classical computation.",
      ) +
      (s === 0
        ? `<div class="scene application-scene">${viz.application(t ?? 1, app)}</div>`
        : `<div class="application-code"><div>${code({ code: app.source, lines_html: app.source_lines_html }, { count: 20, title: "Complete captured PennyLane quantum function", compact: true })}${code({ code: app.batch_source, lines_html: app.batch_source_lines_html }, { count: 6, title: "Actual broadcast submission", compact: true })}</div><div class="application-mini">${viz.application(1, app)}<div class="metric-row">${metric(fmt(app.workload?.snapshots, 0), "programs")}${metric(fmt((app.workload?.snapshots || 0) * (app.workload?.shots_per_snapshot || 0), 0), "quantum shots")}</div></div></div>`) +
      `<div class="reference-strip">${qr("afqmc", "github.com/amazon-braket/amazon-braket-examples", "Public AFQMC example · adapted and verified for LiH")}</div>`,
    "A software workflow demonstration, not a quantum advantage claim. LiH: STO-3G, six active spin orbitals. Quantum measurements produce shadows once. CPUs reuse them for overlap estimates during walker propagation. The second build shows the complete recorded quantum function and its actual broadcast call.",
    { 0: { duration: 6500 } },
  );
  add(
    "afqmc-execution",
    "02 · Quantum and classical execution",
    "Many programs, one hybrid workflow",
    1,
    (s, t = null) =>
      heading(
        "Many programs, one hybrid workflow",
        s
          ? "The same quantum data feeds parallel classical worker tasks."
          : "Submit the program collection, wait, then retrieve each indexed result.",
      ) +
      afqmcExecution(s, t) +
      sourceNote(
        "Recorded local runs · QDMI replay starts at submission · CPU task intervals use their own measured clock",
      ),
    "The replay skips Python circuit preparation and begins at the first QDMI submission. Indexed results arrive at their recorded retrieval times. The second build replays CPU task intervals and imaginary-time walker weights alongside the completed quantum stage. These are different clocks and are labelled separately. Four processes on this machine demonstrate the parallel execution pattern; no HPC cluster execution is claimed.",
    { 0: { duration: 8500 }, 1: { duration: 11000, loop: true } },
  );
  add(
    "afqmc-result",
    "02 · Results and validation",
    "From walker weights to an energy estimate",
    0,
    (s, t = null) =>
      heading(
        "From walker weights to an energy estimate",
        `${esc(app.chemistry.molecule)} · stretched ${fmt(Math.hypot(...app.chemistry.geometry_angstrom[1].map((value, i) => value - app.chemistry.geometry_angstrom[0][i])), 1)} Å bond · ${app.chemistry.electrons} electrons in ${app.chemistry.spin_orbitals} spin orbitals`,
      ) +
      energyChart((t ?? 1) * 3) +
      `<div class="result-bottom"><div class="result-walkers">${walkerScene(t ?? 1)}</div><div><p><b>Blue:</b> quantum-shadow trial<br><b>Dashed:</b> Hartree–Fock trial<br><b>Zero:</b> exact active-space reference</p><p class="note">Band: ±1 walker standard error.<br>Shadow error and systematic biases are separate.</p></div><img class="energy-equation" src="${assets["eq-afqmc-energy"]}" alt="Energy is the importance-weighted walker average"/></div>`,
    "Imaginary time is a projection parameter, not wall time. Circle area is recorded importance weight. The exact reference applies only to this active space; the band is conditional on the shared quantum shadows. Do not imply quantum advantage or scalable dense post-processing.",
    { 0: { duration: 12000, loop: true } },
  );
  add(
    "routing",
    "03 · Compiling for a device",
    "From a circuit to a physical target",
    3,
    (s, t = null) =>
      heading(
        "From a circuit to a physical target",
        "One LiH measurement program · IQM Emerald gate set and connectivity",
      ) +
      routingScene(s, t) +
      `<div class="routing-ref">${qr("iqm", "iqm-finland.github.io/QDMI-on-IQM", "IQM devices through QDMI")}</div>` +
      sourceNote(
        "Captured target model via local SC QDMI · execution on DDSIM · search movement interpolates recorded endpoints",
      ),
    "A simulator accepts the original circuit. A physical target constrains placement, connectivity and native gates. Follow real mapping refinement, routed operations and native synthesis. Counters are compiler introspection, not estimated timings. SWAP count during search refers to the displayed trial. Highlighted operands come from emitted operations.",
    { 1: { duration: 10000 }, 2: { duration: 10000 }, 3: { duration: 8000 } },
  );
  add(
    "device-switch",
    "03 · One device interface",
    "Change the target, recompile the program",
    2,
    (s, t = null) =>
      heading("Change the target, recompile the program") +
      targetSwitchScene(s, t) +
      sourceNote(
        "Captured provider models · local SC compiler target · identical input · unchanged compiled payload executes on DDSIM",
      ),
    "Each forward build selects a different device ID, changes its topology, and replays compilation before revealing the resulting circuit and introspection. IBM is the official public Miami Nighthawk snapshot. IonQ uses fixed equatorial pulses and bounded RZZ interactions; the capture records the exact constraints. No physical hardware jobs are submitted.",
    {
      0: { duration: 5500, transition: 700 },
      1: { duration: 6500, transition: 700 },
      2: { duration: 6500, transition: 700 },
    },
  );
  add(
    "compiler-stages",
    "03 · Inside the MQT Compiler Collection",
    "Inside the compiler",
    3,
    (s) => {
      const ids = ["source", "qc", "qco", "optimized"],
        a = appStage(ids[s]);
      const labels = [
        "OpenQASM input",
        "QC · quantum references",
        "QCO · quantum values",
        "Optimized QCO",
      ];
      const descriptions = [
        "The complete captured circuit enters through OpenQASM.",
        "Mutable quantum references sit alongside standard classical MLIR.",
        "Every operation produces new quantum values: dependencies become explicit.",
        "Canonicalization, cancellation and fusion transform that value graph.",
      ];
      const start = s
        ? focus(a, s === 1 ? "qc.h" : "qco.h", 2)
        : focus(a, "qreg", 0);
      return (
        heading("Inside the compiler") +
        stageFlow(labels, s) +
        `<div class="compiler-caption"><h3>${labels[s]}</h3><p>${descriptions[s]}</p></div><div class="compiler-source">${code(a, { start, count: 17, compact: true, title: `Actual ${a?.label || labels[s]} · consecutive source lines` })}</div><div class="compiler-underlay">${programMetrics(targets[0], ids[s])}<p><b>MLIR</b> reusable infrastructure<br><b>QC / QCO</b> quantum semantics<br><b>arith · scf · func</b> classical semantics</p></div>`
      );
    },
    "The same LiH circuit is shown in a wider consecutive source window. Long generated programs are not represented as complete snippets. Original line numbers identify the view and full artifacts are bundled. Highlight the quantum references becoming SSA values, then the actual optimization changes.",
  );
  add(
    "structured",
    "04 · Structured quantum programs",
    "Preserving quantum program structure",
    4,
    (s) => {
      const labels = [
        "Program & feedback",
        "Native OpenQASM",
        "Adaptive QIR",
        "Bounded unrolling",
        "Repeat until success",
      ];
      let body = "";
      if (s === 0)
        body = `<div class="structured-grid"><div>${code(artifact("source"), { count: 30, title: "Complete OpenQASM 3.1 input", compact: true })}</div><div>${circuit(artifact("source")?.circuit, { limit: 14, key: "structured" })}<p class="statement">Repetition · reset<br>measurement → feedback</p></div></div>`;
      if (s === 1)
        body = `<div class="structured-native"><div>${structureCode(artifact("openqasm3"), "Actual target-native OpenQASM 3.1")}</div><div>${circuit(artifact("source")?.circuit, { limit: 14, key: "structured" })}<p class="statement">Change gates and placement.<br>Preserve loops and feedback.</p><p class="note">All control boundaries are visible.<br>Only gate runs are folded for projection.</p></div></div>`;
      if (s === 2)
        body = structureCode(
          artifact("qir-adaptive"),
          "Actual QIR 2.1 Adaptive Profile · complete entry-point control flow",
          true,
        );
      if (s === 3)
        body = `<div class="split unroll-comparison"><div><span class="tag">Loop retained</span>${circuit(artifact("optimized")?.circuit, { limit: 14, key: "loop-preserved" })}</div><div><span class="tag">Bounded loop expanded</span>${circuit(artifact("unrolled", true)?.circuit || artifact("optimized", true)?.circuit, { limit: 20, key: "loop-expanded" })}</div></div><p class="statement">Both outputs still need measurement-dependent feedback.</p>`;
      if (s === 4) {
        const v = variant("rus"),
          a = v.stages.find((a) => a.id === "source");
        body = `<div class="structured-grid"><div>${code(a, { count: 30, title: "Complete repeat-until-success program", compact: true })}</div><div>${circuit(a.circuit, { limit: 14, key: "rus" })}<p class="statement">A runtime condition determines<br>how often the body executes.</p><p class="note">The actual compiled QIR executes through DDSIM.</p></div></div>`;
      }
      return (
        heading("Preserving quantum program structure") +
        stageFlow(labels, s) +
        body +
        `<div class="reference-strip">${qr("unrolling", "arxiv.org/abs/2609.16171", "Why are we unrolling? · structured quantum compilation")}</div>`
      );
    },
    "A small recognizable syndrome-extraction pattern replaces the large static chemistry circuit. Read its complete source. Then retain all loop, reset, measurement and feedback boundaries in native output. Display folds are labelled and do not modify source. Adaptive QIR retains the control-flow graph. Bounded unrolling cannot remove genuine measurement-dependent feedback. Repeat-until-success illustrates why runtime structure matters.",
  );
  add(
    "adaptive-execution",
    "04 · Adaptive QIR execution",
    "Adaptive execution through QDMI",
    0,
    (s, t = null) =>
      heading(
        "Adaptive execution through QDMI",
        "Iterative QPE · four logical qubits · eight output bits · phase 1/3",
      ) +
      executionScene(
        (t ?? 1) * totalTime(),
        "Measured timeline · slowed for inspection",
      ),
    "One moderate-speed replay replaces the fast and slow versions. Follow API calls and shot completions on the same recorded clock. The non-exact phase gives a real distribution over eight output bits. Every histogram increment corresponds to recorded shot completion.",
    { 0: { duration: 9500 } },
  );
  add(
    "mqt-core",
    "05 · The software behind the workflow",
    "MQT Core",
    0,
    () =>
      heading(
        "MQT Core",
        "Open infrastructure for quantum programs, compilers and execution.",
      ) +
      `<div class="scene core-world" data-morph="world">${viz.architecture("core")}</div><div class="core-impact-row"><div class="metric-row">${metric(references.core_stats?.github?.stargazers_count ?? "—", "GitHub stars")}${metric(references.core_stats?.dashboard_snapshot?.total_downloads ?? "—", "PyPI downloads", true)}</div>${qr("core-repo", "github.com/munich-quantum-toolkit/core", "Source & contributions")}${qr("core", "mqt.readthedocs.io/projects/core", "Documentation & examples")}</div>`,
    "Zoom out from the demonstrated paths to the library: quantum IR, decision diagrams, ZX, languages, compilation, QDMI and execution. Stars and cumulative downloads are recorded snapshots, not unique-user counts.",
  );
  add(
    "ecosystem",
    "05 · From the metal to the user",
    "System software for quantum computing",
    0,
    () =>
      heading("System software for quantum computing") +
      `<div class="scene core-world closing-world" data-morph="world">${viz.architecture("ecosystem")}</div><div class="closing-resources"><div class="closing-logos">${logo("mqt")}${logo("qdmi")}${logo("mqss")}${logo("mqv")}${logo("tum-cda")}</div>${qr("company", "mq.sc", "From the Metal to the User")}</div>`,
    "Let Core morph into the larger system. Users, classical infrastructure and quantum devices connect through a shared stack. End with practical hybrid computation, device-aware compilation and preserved quantum program structure. Credit the participating projects and TUM colleagues.",
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
    $("slide-number").innerHTML =
      `${String(state.slide + 1).padStart(2, "0")} / ${slides.length}<span class="build-dots" aria-label="Build ${state.step + 1} of ${slide.builds + 1}">${Array.from({ length: slide.builds + 1 }, (_, i) => (i === state.step ? "<b>•</b>" : "•")).join("")}</span>`;
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
      const loop = state.animation.loop;
      state.animation.end();
      if (!loop) return;
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
      .map(
        (s, i) =>
          `<section class="print-slide"><header class="slide-header"><span>${esc(s.section)}</span>${logo("mqsc")}</header><article>${s.render(s.builds)}</article><footer class="slide-footer"><span>Lukas Burgholzer · MQSC</span><span>mq.sc</span><span>${i + 1} / ${slides.length}</span></footer></section>`,
      )
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
