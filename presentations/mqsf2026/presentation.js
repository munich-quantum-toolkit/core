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
  const reveal = (step, at, html) =>
    `<div class="reveal ${step < at ? "off" : ""}" aria-hidden="${step < at}">${html}</div>`;
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
    {
      start = 0,
      count = 11,
      hot = [],
      title,
      compact = false,
      columns = 96,
    } = {},
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
        const content =
          l.length > columns
            ? esc(l.slice(0, columns)) + '<span class="elision"> …</span>'
            : (highlighted ?? esc(l));
        return `<span class="code-line ${hot.includes(start + i) ? "hot" : ""}"><span class="line-no">${start + i + (a.excerpt_start_line || 1)}</span><span class="code-text">${content}</span></span>`;
      })
      .join("");
    return `<div data-morph="code-panel" class="code-frame ${compact ? "compact" : ""}"><div class="code-title">${esc(title || a.label || a.language || "Actual source")}</div><pre class="highlight">${body}</pre>${shown.some((l) => l.length - indent > columns) ? '<p class="code-foot">… Excerpt shortened for projection; full source in the bundle.</p>' : ""}</div>`;
  }
  const focus = (a, needle, before = 1) =>
    Math.max(
      0,
      (a?.code?.split("\n").findIndex((s) => s.includes(needle)) ?? 0) - before,
    );
  const flatten = window.MQSF_CIRCUIT.flatten;
  const circuit = window.MQSF_CIRCUIT.render;
  function pipeline(step) {
    return `<div class="pipeline">${["OpenQASM", "QC", "QCO", "Optimize", "Target"].map((s, i) => `${i ? '<span class="arrow">→</span>' : ""}<span class="stage ${i === step ? "active" : ""}">${s}</span>`).join("")}</div>`;
  }
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
    const events = capture.events || [],
      e = [...events].reverse().find((e) => e.time_ms <= elapsed),
      a = {
        code: capture.client_source || "",
        lines_html: capture.client_source_lines_html,
        label:
          capture.trace_kind === "python-observed-adapter-calls"
            ? "Actual Core PennyLane adapter"
            : "Actual client · Python → QDMI C ABI",
      };
    const srcLine = e?.source_line || 1;
    return code(a, {
      start: Math.max(0, srcLine - 2),
      count: 3,
      columns: 42,
      hot: [srcLine - 1],
      compact: true,
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
  function applicationFlow(step) {
    const boxes = [
      [
        55,
        95,
        340,
        165,
        "Molecular problem",
        "Integrals · trial wavefunction",
        "molecule",
        0,
      ],
      [
        590,
        95,
        390,
        165,
        "Quantum sampling",
        `${fmt(app.workload?.snapshots, 0)} shadow circuits · ${fmt(app.workload?.shots_per_snapshot, 0)} shots`,
        "circuit",
        1,
      ],
      [
        1210,
        95,
        330,
        165,
        "Classical shadows",
        "Basis + measured bitstrings",
        "measurement",
        1,
      ],
      [
        590,
        405,
        390,
        165,
        "Parallel AFQMC",
        `${fmt(app.propagation?.walkers, 0)} walkers · ${fmt(app.propagation?.steps, 0)} steps`,
        "server",
        2,
      ],
      [
        1210,
        405,
        330,
        165,
        "Energy estimate",
        "Weighted ensemble average",
        "orbitals",
        3,
      ],
    ];
    let body = `<path class="flow-track" d="M395 178H590M980 178H1210M1375 260V338H785V405M980 487H1210" fill="none" stroke="#a6c3de" stroke-width="4"/>`;
    body += `<path class="flow-particles" d="M395 178H590M980 178H1210M1375 260V338H785V405M980 487H1210" fill="none" stroke="#2f70b8" stroke-width="6" stroke-dasharray="12 105"/>`;
    body += boxes
      .map(
        ([x, y, w, h, title, sub, icon, at], i) =>
          `<g data-morph="workflow-${i}" opacity="${step >= at ? 1 : 0.12}" transform="translate(${x} ${y})"><rect width="${w}" height="${h}" rx="18" fill="${i === 3 ? "#e4f1fb" : "#f3f7fc"}" stroke="#b9cee2" stroke-width="2"/>${viz.icon(icon, w / 2 - 31, 12, 62)}${text(w / 2, 105, title, 28)}${text(w / 2, 140, sub, 20, "#526d88")}</g>`,
      )
      .join("");
    body += `<g data-morph="workflow-reuse" opacity="${step >= 2 ? 1 : 0}"><path d="M630 570C430 675 410 405 590 445" fill="none" stroke="#087d92" stroke-width="3"/><path d="M577 435L593 445L574 451" fill="none" stroke="#087d92" stroke-width="3"/>${text(270, 550, "Propagate → overlap → reweight", 24, "#087d92")}${text(270, 588, "Reuse quantum data at every step", 22, "#526d88")}</g>`;
    body +=
      text(80, 32, "Quantum stage", 24, "#2f70b8", "start") +
      text(80, 376, "Classical compute", 24, "#2f70b8", "start");
    return svg(
      body,
      "0 0 1600 650",
      "Quantum-assisted AFQMC algorithm and classical parallel propagation",
    );
  }
  function walkerScene(progress = 1) {
    const p = app.propagation,
      frames = p?.curves?.[0]?.frames || [];
    if (!frames.length) return "";
    const index = Math.min(frames.length - 1, progress * (frames.length - 1)),
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
    const elapsed =
      progress === null ? (step ? duration : 0) : progress * duration;
    const submit = events.find((e) => e.operation.includes("try_submit_job"));
    const preparing = elapsed < (submit?.time_ms ?? 0);
    const client = preparing
      ? code(
          {
            code: app.batch_source,
            lines_html: app.batch_source_lines_html,
            label: "Actual PennyLane broadcast call",
          },
          { count: 3, columns: 57, compact: true },
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
    return `<div class="execution-grid batch-grid" data-programs="${count}"><div>${client}<p class="batch-phase">${preparing ? "Prepare and lower the circuit collection" : count === snapshots.length ? "All indexed samples retrieved" : "Submit → wait → retrieve by program index"}</p>${sequence(capture, elapsed)}</div><div><div class="runtime-strip"><span>Recorded call · slowed replay</span><span>${fmt(elapsed, 1)} / ${fmt(duration, 1)} ms</span></div><div class="clock-line"><div style="width:${(elapsed / duration) * 100}%"></div></div><div class="batch-result">${svg(body, "0 0 750 420", "Captured indexed program results arriving at their actual retrieval times")}</div><div class="metric-row">${metric(fmt(app.workload?.snapshots, 0), "programs")}${metric(fmt(count * (app.workload?.shots_per_snapshot || 0), 0), "shots retrieved")}</div></div></div>`;
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
    if (fidelities.size)
      body +=
        text(410, 590, "Reported 2Q fidelity", 23, "#526d88") +
        text(190, 622, "< 98%", 21, "#ce9152") +
        text(410, 622, "98–99.5%", 21, "#54a0b2") +
        text(630, 622, "≥ 99.5%", 21, "#2f70b8");
    else
      body += text(
        410,
        620,
        "All-to-all connectivity · aggregate calibration",
        22,
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
      names = (m.operations || []).map((o) =>
        typeof o === "string" ? o : o.name,
      );
    return `<div class="device-properties"><div><b>${m.qubits || m.sites?.length || 0}</b><span>qubits</span></div><div><b>${m.edges?.length || 0}</b><span>couplings</span></div><div class="gate-set"><b>${esc(names.join(" · "))}</b><span>native operations</span></div></div>`;
  }
  function calibrationSummary(target) {
    const c = target.metadata?.calibration || {};
    return (
      [
        ["one_qubit", "1Q"],
        ["two_qubit", "2Q"],
        ["readout", "Readout"],
      ]
        .map(
          ([key, label]) =>
            `${label}: ${c[key]?.available ? fmt(c[key].mean * 100, 2) + "%" : "unavailable"}`,
        )
        .join(" · ") +
      `<br>Reported means · ${esc((target.metadata?.calibration_date || "undated").slice(0, 10))}`
    );
  }
  function routingScene(step, progress = null) {
    const target = targets[0],
      comp = target?.compilation || {},
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
        ? appStage("optimized")?.circuit || appCircuit()
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
    return `<div class="architecture-grid"><div><div class="phase-labels">${["Placement", "Forward ↔ backward", "Routing", "Native synthesis"].map((l, i) => `<span class="${i === step ? "selected" : ""}">${l}</span>`).join("")}</div>${circuit(circuitData, { active: opIndices, camera: step >= 2 ? layerPosition : null, limit: 18, key: step < 2 ? "routing-logical" : step === 2 ? "routing-routed" : "routing-native" })}<div class="mapping-legend">${step === 3 ? "Physical wires shown in the circuit" : placement.map((p, i) => `<span>q${i} → ${p}</span>`).join("")}</div><div class="telemetry"><b>${phase}</b><span>${step === 1 ? `Search event ${index + 1} / ${search.length}` : step === 3 ? nativeSummary : "Actual compiler output"}</span>${step < 3 ? `<strong>${step === 1 ? (event.swaps ?? events.find((e) => e.phase === "score" && e.trial === event.trial)?.swaps ?? 0) : step === 2 && progress !== null ? swaps : (comp.layout?.swaps?.length ?? 0)}<small> SWAPs</small></strong>` : ""}</div></div><div>${deviceView(target, active, step === 3 ? null : placement, blend, step >= 1 ? activeEdges : null)}</div></div>`;
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
    return `<div class="hpc-lanes">${svg(body, "0 0 1550 385", "Actual four-process AFQMC task intervals on a measured wall clock")}</div><div class="hpc-details"><div>${code(src, { start: focus(src, "for label in", 0), count: 5, columns: 74, compact: true, title: "Actual CPU task submission" })}</div><div><div class="metric-row">${metric(p.processes || 0, "CPU processes")}${metric(tasks.length, "walker tasks")}</div><p class="note">Each bar: one complete ${p.steps}-step walker.<br>Blue: shadow trial · grey: Hartree–Fock trial.<br>Measured steady-phase zoom; whole pool: ${fmt((p.duration_ms || 0) / 1000, 2)} s, including startup.</p></div></div>`;
  }
  function coreStack(step) {
    const layers = [
      ["Frontends & exchange", "OpenQASM 3.1 · Qiskit · jeff", 0],
      [
        "MQT Compiler Collection",
        "MLIR · QC / QCO · placement · routing · synthesis",
        1,
      ],
      [
        "Program representations & libraries",
        "Quantum IR · decision diagrams · ZX calculus",
        1,
      ],
      [
        "QDMI integration",
        "Device discovery · batching · Qiskit / PennyLane adapters",
        2,
      ],
      [
        "Execution & verification",
        "DDSIM · structured programs · OpenQASM / QIR",
        2,
      ],
    ];
    let body = layers
      .map(
        ([title, sub, at], i) =>
          `<g data-morph="core-layer-${i}" opacity="${step >= at ? 1 : 0.08}" transform="translate(${step < at ? 55 : 0} ${i * 102})"><rect x="15" y="0" width="1010" height="87" rx="13" fill="${i === 1 ? "#2f70b8" : "#e9f2fa"}"/>${text(43, 35, title, 30, i === 1 ? "white" : "#142b45", "start")}${text(43, 67, sub, 22, i === 1 ? "#eaf2fa" : "#526d88", "start")}</g>`,
      )
      .join("");
    return svg(
      body,
      "0 0 1045 515",
      "MQT Core components arranged as a quantum software stack",
    );
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
      `<div class="title-layout"><h1>System Software for Quantum Computing:<br><span class="blue">From the Metal to the User</span></h1><div class="title-byline"><b>Lukas Burgholzer</b><span>CTO &amp; Co-founder · MQSC</span><span>Technical University of Munich</span></div><div class="title-animation" data-morph="world">${viz.bridge(3)}</div></div>`,
    "The application, the computing infrastructure, and the quantum device need to work together. Introduce MQSC as the system software company connecting these layers.",
  );
  add(
    "integration",
    "01 · System software",
    "Hardware scales. Software breaks.",
    2,
    (s) =>
      heading('Hardware scales.<span class="blue"> Software breaks.</span>') +
      `<div class="scene ecosystem-scene" data-morph="world">${viz.ecosystem(s)}</div>`,
    "Build the people who need quantum systems, the heterogeneous hardware and classical computing resources, then the bespoke integrations. These connections are architectural illustrations, not measurements of integration effort.",
  );
  add(
    "shared-stack",
    "01 · System software",
    "A shared software stack",
    0,
    () =>
      heading("A shared software stack") +
      `<div class="scene ecosystem-scene" data-morph="world">${viz.ecosystem(3)}</div>`,
    "Let the same objects move into place. MQSC and MQT supply the shared software between users, classical compute, and quantum hardware.",
  );
  add(
    "architecture",
    "01 · System software",
    "From the metal to the user",
    3,
    (s) =>
      heading("From the metal to the user") +
      `<div class="scene stack-scene" data-morph="world">${viz.stack(s)}</div><div class="reference-corner">${qr("mqss", "doi.org/10.1145/3773656.3773669", "Munich Quantum Software Stack · SCA/HPCAsia 2026")}</div>`,
    "Zoom into the middle. Frontends and backends are separate; compilation connects program semantics to hardware constraints; resource management connects execution to classical computing. The paper describes the broader architecture, not only Core.",
  );
  add(
    "chemistry",
    "02 · A hybrid application",
    "Quantum-assisted auxiliary-field Monte Carlo",
    2,
    (s) =>
      heading(
        "Quantum-assisted auxiliary-field Monte Carlo",
        "Estimate molecular ground-state energies using quantum and classical computation.",
      ) +
      `<div class="scene chemistry-scene">${viz.molecule(s)}</div><div class="projection-formula"><span>Imaginary-time projection</span><img src="${assets["eq-afqmc-projection"]}" alt="Ground state from imaginary-time propagation of the trial state" /></div>` +
      sourceNote(
        `${esc(app.chemistry?.molecule || "LiH")} · ${esc(app.chemistry?.basis || "STO-3G")} · active space: 2 electrons / 3 spatial orbitals · ideal DDSIM simulation`,
      ),
    "LiH provides a concrete near-term application. Define the active space before showing circuits. The six-qubit trial was tuned classically for this verified small demonstration; this is a workflow demonstration, not evidence of quantum advantage.",
  );
  add(
    "hybrid-algorithm",
    "02 · A hybrid application",
    "The quantum–classical algorithm",
    3,
    (s) =>
      heading("The quantum–classical algorithm") +
      `<div class="scene workflow-scene">${applicationFlow(s)}</div>` +
      sourceNote(
        "Quantum measurements supply trial overlaps. Classical processors propagate and reweight the walker ensemble.",
      ),
    "Read the full algorithm. Generate a collection of randomized measurement programs. Submit them as one multi-program job. Reuse those shadows for overlap estimates throughout imaginary-time propagation; there is no invented quantum call inside every walker update.",
  );
  add(
    "afqmc-execution",
    "02 · Device execution through QDMI",
    "Many programs, one device job",
    2,
    (s, t = null) =>
      heading(
        "Many programs, one device job",
        `${fmt(app.workload?.snapshots, 0)} shadow programs × ${fmt(app.workload?.shots_per_snapshot, 0)} shots · native QDMI batching`,
      ) +
      batchScene(s, t) +
      sourceNote(
        "Recorded local DDSIM run · API events use measured timestamps · results appear when they are retrieved",
      ),
    "The actual submission source and QDMI call trace share one clock. First follow the recorded call; then review the complete batch. The square array represents the collection, not a per-program completion clock. Use the retained payloads and indexed results for debugging.",
    { 1: { duration: 12000 } },
  );
  add(
    "classical-compute",
    "02 · Classical compute",
    "Parallel propagation of the walker ensemble",
    2,
    (s, t = null) =>
      s === 2
        ? heading(
            "Classical parallelism in the same workflow",
            "Four local processes execute independent walker tasks and reduce their results.",
          ) + hpcScene(t ?? 1)
        : heading(
            "Parallel propagation of the walker ensemble",
            "Each walker is a Slater determinant—a numerical electronic state.",
          ) +
          `<div class="split walker-layout"><div class="walker-scene">${walkerScene(t === null ? (s ? 1 : 0) : t)}</div><div><p class="eyebrow">Classical AFQMC</p><ol class="algorithm-steps"><li>Propagate in imaginary time</li><li>Evaluate trial overlaps</li><li>Apply the phaseless constraint</li><li>Reweight and reduce energies</li></ol><div class="metric-row">${metric(app.propagation?.walkers || 0, "walkers")}${metric(app.propagation?.steps || 0, "time steps")}</div><p class="note">Circle area tracks recorded walker weight.<br>Imaginary time is a projection parameter.<br>Interpolation only smooths the replay.</p></div></div>`,
    "Explain the ensemble before starting its movement. Imaginary time is a mathematical projection parameter, not wall time or a molecular trajectory. Quantum-derived trial overlaps guide classical matrix operations. Independent walker partitions are processed in parallel and reduced to the weighted estimate.",
    { 1: { duration: 20000, loop: true }, 2: { duration: 10000, loop: true } },
  );
  add(
    "afqmc-result",
    "02 · Results and validation",
    "From the walker ensemble to an energy estimate",
    2,
    (s, t = null) =>
      heading(
        "From the walker ensemble to an energy estimate",
        "Quantum-derived trial overlaps guide a classical imaginary-time projection.",
      ) +
      energyChart(t === null ? (s === 0 ? 0 : 3) : t * 3) +
      `<div class="result-notes"><p><b>Blue:</b> classically tuned, shadow-derived trial &nbsp; <b>Dashed:</b> Hartree–Fock trial<br><b>Zero:</b> exact active-space reference</p><p>Band: ±1 walker standard error.<br>Shadow error and systematic biases are separate.</p></div>` +
      sourceNote(
        `${esc(app.chemistry?.molecule || "LiH")} active space · exact Hamiltonian / overlap checks · full recorded trajectory`,
      ),
    "Grow the measured trajectory. The energy estimate is a weighted reduction over walkers, and each time step reuses the quantum measurement data. Explain the band and reference; do not imply the finite active-space calculation establishes chemical accuracy for the full molecule.",
    { 1: { duration: 16000, loop: true } },
  );
  add(
    "device-contract",
    "03 · Compiling for a device",
    "A simulator is only the beginning",
    1,
    (s) =>
      heading(
        "A simulator is only the beginning",
        "A physical target defines where operations can run and which gates are available.",
      ) +
      `<div class="architecture-grid"><div><div class="device-id">device_id = <b>"iqm.emerald"</b></div>${targetSummary(targets[0])}<div class="device-questions"><p>Connectivity → placement and routing</p><p>Native operations → synthesis</p><p>Control capabilities → legal program forms</p></div>${s ? qr("iqm", "iqm-finland.github.io/QDMI-on-IQM", "IQM devices through QDMI") : logo("qdmi")}</div><div>${deviceView(targets[0])}</div></div>` +
      sourceNote(
        "Provider metadata captured before the talk · SC target adapter for compilation · compiled payloads execute on DDSIM",
      ),
    "The same workflow first ran on a simulator. Targeting real hardware adds constraints. The Emerald graph and calibration are retrieved metadata, the local compiler target is an SC adapter, and no job is submitted to hardware.",
  );
  add(
    "compiler-stages",
    "03 · MQT Compiler Collection",
    "A program through the compiler",
    4,
    (s) => {
      const ids = [
          "source",
          "qc",
          "qco",
          "optimized",
          "target-native-synthesis",
        ],
        a = appStage(ids[s]) || artifact(ids[s]);
      const labels = [
        "OpenQASM input",
        "QC: quantum references",
        "QCO: explicit quantum values",
        "Optimize quantum operations",
        "Target-native program",
      ];
      const descriptions = [
        "Import gates, types and classical computations into MLIR.",
        "Operations act on references; classical code uses standard MLIR dialects.",
        "Single-use quantum values make dependencies explicit for transformations.",
        "Simplify and combine operations while respecting their quantum semantics.",
        "Placement, routing and synthesis use the selected QDMI device.",
      ];
      const needle =
        s === 0 ? "qreg" : s === 1 ? "qc.h" : s === 4 ? "qco.r(" : "qco.h";
      return (
        heading("A program through the compiler") +
        pipeline(s) +
        `<div class="compiler-caption"><h3>${labels[s]}</h3><p>${descriptions[s]}</p></div><div class="compiler-source">${code(a, { start: focus(a, needle, 0), count: 8, compact: true, title: a?.label || labels[s] })}</div><div class="dialect-key"><span><b>MLIR</b> reusable compiler infrastructure</span><span><b>QC / QCO</b> quantum semantics</span><span><b>arith / scf / func</b> classical semantics</span></div>`
      );
    },
    "Follow one captured LiH measurement program. Explain what to look at in each form; do not read every operand. QCO names successive quantum values explicitly. Classical MLIR remains present when the input requires it.",
  );
  add(
    "routing",
    "03 · Device-aware compilation",
    "Placement, routing and native gates",
    3,
    (s, t = null) =>
      heading("Placement, routing and native gates") +
      routingScene(s, t) +
      sourceNote(
        "Recorded traversal endpoints; movement interpolated · emitted gate operands in the routing replay · local SC target",
      ),
    "Place logical qubits, replay actual forward and backward search traversals, then walk the routed circuit beside its physical operands. SWAP counts refer to the displayed compiler traversal or selected output. Native synthesis follows routing; the diagram is derived from captured operations.",
    { 1: { duration: 18000 }, 2: { duration: 16000 }, 3: { duration: 14000 } },
  );
  add(
    "device-switch",
    "03 · One device interface",
    "Change the target, recompile the program",
    2,
    (s) => {
      const target = targets[Math.min(s, targets.length - 1)] || {},
        a = targetStage(target, "openqasm3");
      const firstGate = Math.max(
        0,
        (a?.code || "").split("\n").findIndex((l) => /^[a-z].*\$\d/.test(l)),
      );
      return (
        heading("Change the target, recompile the program") +
        `<div class="architecture-grid target-switch"><div><div class="device-id" data-morph="device-id">device_id = <b>"${esc(target.id || "iqm.emerald")}"</b></div><h3>${esc(target.label || "IQM Emerald")}</h3>${targetSummary(target)}${code(a, { start: firstGate, count: 6, compact: true, columns: 49, title: s === 2 ? "Actual QIS interface OpenQASM" : "Actual target-native OpenQASM" })}<p class="calibration">${calibrationSummary(target)}</p></div><div>${deviceView(target, target.compilation?.layout?.initial || [], target.compilation?.layout?.initial)}<p class="target-status">${esc(target.metadata?.basis_note || "Captured provider metadata; compilation uses local SC target.")}</p></div></div>` +
        sourceNote(
          "Identical input program · target-specific gate set and topology · execution on DDSIM; no hardware submissions",
        )
      );
    },
    "Switch the ID from IQM to the authentic public IBM Nighthawk snapshot, then IonQ. Device metadata, topology and actual compiled program change together. The IBM calibration is dated; these are offline captures, not live network queries.",
  );
  add(
    "structured",
    "04 · Beyond static circuits",
    "Structured quantum programs",
    3,
    (s) =>
      heading(
        "Structured quantum programs",
        "Syndrome extraction combines repetition, reset and measurement feedback.",
      ) +
      `<div class="structured-grid"><div>${code(artifact("source"), { count: 12, hot: s === 1 ? [5, 10] : s === 2 ? [8, 9] : [], title: "Actual OpenQASM 3.1 input", compact: true })}</div><div>${circuit(artifact("source")?.circuit, { active: [-1, 2, 5, 6][s], limit: 12, key: "structured" })}<div class="concepts"><p class="${s >= 1 ? "selected" : ""}">Bounded repetition</p><p class="${s >= 2 ? "selected" : ""}">Measurement → conditional correction</p><p class="${s >= 3 ? "selected" : ""}">Structured control survives compilation</p></div></div></div>`,
    "Move from the near-term chemistry circuit to a small recognizable feedback program for fault-tolerant workloads. This is a toy syndrome-extraction pattern, not a complete error-correcting code. The meter feeds the conditional gate; the loop body is shown once.",
  );
  add(
    "preserve-structure",
    "04 · Structure through compilation",
    "Preserve structure all the way to the executable",
    4,
    (s) => {
      if (s === 4)
        return (
          heading("Expand bounded loops when a target requires it") +
          `<div class="split unroll-comparison"><div><span class="tag">Structured program</span>${circuit(artifact("optimized")?.circuit, { limit: 14, key: "loop-preserved" })}</div><div><span class="tag">Bounded loop unrolled</span>${circuit(artifact("unrolled", true)?.circuit || artifact("optimized", true)?.circuit, { limit: 20, key: "loop-expanded", label: "Two rounds expanded · measurement feedback retained" })}</div></div><p class="statement unroll-statement">Measurement-dependent corrections<br>remain adaptive in both outputs.</p><div class="reference-corner">${qr("unrolling", "arxiv.org/abs/2609.16171", "Why are we unrolling? · structured compilation")}</div>`
        );
      const ids = [
          "qco",
          "target-native-synthesis",
          "openqasm3",
          "qir-adaptive",
        ],
        a = artifact(ids[s]),
        needle =
          s < 2 ? "scf.for" : s === 2 ? "for " : "__quantum__rt__read_result";
      return (
        heading("Preserve structure all the way to the executable") +
        `<div class="format-tabs">${["QCO", "Native QCO", "OpenQASM 3.1", "QIR 2.1 Adaptive"].map((l, i) => `<span class="${i === s ? "selected" : ""}">${l}</span>`).join("")}</div><div class="compiler-source">${code(a, { start: focus(a, needle, 0), count: 9, compact: true, title: a?.label })}</div><div class="dialect-key"><span><b>Keep</b> loops and measurement feedback</span><span><b>Change</b> gates and physical placement</span><span><b>Execute</b> through the DDSIM QDMI device</span></div>`
      );
    },
    "Each excerpt is compiler output for the same program. Both supported output formats retain the required adaptive control. QIR 2.1 Adaptive Profile includes the classical computations and dynamic features needed by supported programs. Broad support is not a claim that every construct is supported.",
  );
  add(
    "dynamic-primitives",
    "04 · Dynamic programs",
    "Loops, feedback and repeat-until-success",
    2,
    (s) => {
      const rus = variant("rus"),
        a = [...(rus.stages || []), ...(rus.exports || [])].find(
          (a) => a.id === (s === 2 ? "qir-adaptive" : "source"),
        );
      return (
        heading("Loops, feedback and repeat-until-success") +
        `<div class="dynamic-grid"><div>${code(a, { start: s === 2 ? Math.max(0, focus(a, "br i1", 0) - 2) : 0, count: 11, compact: true, columns: 49, title: s === 2 ? "Actual Adaptive QIR" : "Actual repeat-until-success input" })}</div><div><div class="dynamic-cycle">${svg(`<path d="M120 190C120 40 580 40 580 190C580 345 120 345 120 190" fill="none" stroke="#bad0e5" stroke-width="7"/><path class="flow-particles" d="M120 190C120 40 580 40 580 190C580 345 120 345 120 190" fill="none" stroke="#2f70b8" stroke-width="7" stroke-dasharray="20 180"/>${text(350, 155, "Attempt → measure", 38)}${text(350, 220, "Retry until success", 34, "#2f70b8")}`, "0 0 700 375", "Repeat-until-success control loop")}</div><p class="statement">A runtime condition<br>cannot be statically expanded.</p>${s >= 1 ? qr("unrolling", "arxiv.org/abs/2609.16171", "Why are we unrolling? · structured quantum compilation") : ""}</div></div>`
      );
    },
    "Distinguish a known trip count from a measurement-dependent loop. If a target lacks counted iteration, a bounded loop can be expanded; this does not remove genuine adaptive feedback. The repeat-until-success source and QIR are actual artifacts.",
  );
  add(
    "adaptive-execution",
    "04 · Executing adaptive QIR",
    "Adaptive execution through QDMI",
    2,
    (s, t = null) =>
      heading(
        "Adaptive execution through QDMI",
        "Iterative QPE · four logical qubits · eight output bits · 2,048 shots",
      ) +
      (s === 0
        ? `<div class="qpe-intro"><div><p class="eyebrow">Estimate a non-exact binary phase</p><img src="${assets["eq-qpe-phase"]}" alt="Eigenphase relation and its finite binary estimate"/><p>Eight measured bits resolve the phase on a grid.<br>Repeated shots reveal a distribution around 1/3.</p><div class="metric-row">${metric("QIR 2.1", "Adaptive Profile")}${metric("DDSIM", "QDMI device")}</div></div><div>${histogram(run)}</div></div>`
        : executionScene(
            t === null ? totalTime() : t * totalTime(),
            s === 1
              ? "Recorded time · 1×"
              : "Measured timeline · slowed for inspection",
          )),
    "Use a richer execution workload to make the distribution visible. First play the observed runtime, then the same trace slowed down. Measurement counts change only at real recorded shot-completion timestamps. No hardware is contacted.",
    { 1: { duration: totalTime() }, 2: { duration: 18000 } },
  );
  add(
    "mqt-core",
    "05 · The software behind the workflow",
    "MQT Core",
    3,
    (s) =>
      heading(
        "MQT Core",
        "Open infrastructure for quantum programs, compilers and execution.",
      ) +
      `<div class="core-overview"><div class="core-stack">${coreStack(s)}</div><div>${logo("mqt")}<div data-morph="core-impact" class="core-impact ${s >= 3 ? "" : "muted-impact"}"><div class="metric-row">${metric(references.core_stats?.github?.stargazers_count ?? "—", "GitHub stars")}${metric(references.core_stats?.dashboard_snapshot?.total_downloads ?? "—", "PyPI downloads", true)}</div><p class="note">Dashboard snapshot: 9 October 2026 · downloads, not users</p><div class="core-qr">${qr("core-repo", "github.com/munich-quantum-toolkit/core", "Source & contributions")}${qr("core", "mqt.readthedocs.io/projects/core", "Documentation & examples")}</div></div></div></div>`,
    "Zoom out from the demonstrated compiler and DDSIM into the whole Core. The stack also includes IR, decision diagrams, ZX, language integrations and device adapters. Stars and cumulative downloads are sourced snapshots, not adoption claims about unique users.",
  );
  add(
    "ecosystem",
    "05 · From the metal to the user",
    "System software for quantum computing",
    1,
    (s) =>
      heading("System software for quantum computing") +
      `<div class="scene closing-stack" data-morph="world">${viz.ecosystem(3)}</div><div class="closing-resources">${s ? `${qr("company", "mq.sc", "From the Metal to the User")}${qr("compiler", "mqt.readthedocs.io/projects/core", "Try the compiler and device interfaces")}` : `<div class="closing-logos">${logo("mqt")}${logo("qdmi")}${logo("mqss")}${logo("mqv")}${logo("tum-cda")}</div>`}</div>`,
    "Return to the architecture with evidence behind every layer: a complete hybrid application, portable device access, hardware-aware compilation and preserved dynamic structure. Credit MQT, QDMI, MQSS, Munich Quantum Valley and TUM CDA. Leave the short URL and QR codes visible for discussion.",
  );

  function resize() {
    document.documentElement.style.setProperty(
      "--scale",
      Math.min(innerWidth / 1920, innerHeight / 1080),
    );
  }
  function render(progress = null) {
    state.token++;
    state.animation = null;
    draw(progress, true);
  }
  function draw(progress = null, morph = false) {
    const slide = slides[state.slide];
    $("section-label").textContent = slide.section;
    window.MQSF_MOTION.replace(
      $("slide"),
      slide.render(state.step, progress),
      morph ? 1000 : 0,
    );
    $("slide").dataset.slide = String(state.slide + 1);
    $("slide").dataset.step = String(state.step);
    $("slide").dataset.progress = progress === null ? "" : String(progress);
    $("slide-number").innerHTML =
      `${String(state.slide + 1).padStart(2, "0")} / ${slides.length}<span class="build-dots" aria-label="Build ${state.step + 1} of ${slide.builds + 1}">${Array.from({ length: slide.builds + 1 }, (_, i) => (i === state.step ? "<b>•</b>" : "•")).join("")}</span>`;
    $("build-progress").style.width =
      `${((state.slide + (state.step + 1) / (slide.builds + 1)) / slides.length) * 100}%`;
    $("speaker-notes").textContent =
      `${state.slide + 1}. ${slide.title} — ${slide.notes}`;
    history.replaceState(null, "", `#${state.slide + 1}.${state.step}`);
  }
  function animateTimeline(playback) {
    const token = state.token,
      begin = performance.now(),
      duration = playback.duration;
    let last = 0;
    const end = () => {
      if (token !== state.token) return;
      state.animation = null;
      draw(null, false);
    };
    state.animation = { end, loop: !!playback.loop };
    function frame(now) {
      if (token !== state.token || !state.animation) return;
      const elapsed = now - begin,
        cycle = playback.loop ? elapsed % (duration + 2400) : elapsed;
      const progress = Math.min(1, cycle / duration);
      if (now - last > 30 || progress === 1) {
        draw(progress, false);
        last = now;
      }
      if (playback.loop || elapsed < duration) requestAnimationFrame(frame);
      else end();
    }
    requestAnimationFrame(frame);
  }
  function go(slide, step = 0) {
    state.slide = Math.max(0, Math.min(slide, slides.length - 1));
    state.step = Math.max(0, Math.min(step, slides[state.slide].builds));
    render();
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
      const playback = current.playbacks[state.step];
      render(playback ? 0 : null);
      if (playback) animateTimeline(playback);
    } else if (state.slide < slides.length - 1) go(state.slide + 1);
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
