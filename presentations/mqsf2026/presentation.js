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
    `<h2>${title}</h2>${sub ? `<p class="subtitle">${sub}</p>` : ""}`;
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
    start = Math.max(0, Math.min(start, Math.max(0, lines.length - count)));
    const shown = lines.slice(start, start + count);
    return `<div class="code-frame ${compact ? "compact" : ""}"><div class="code-title">${esc(title || a.label || a.language || "Actual source")}</div><pre class="highlight">${shown.map((l, i) => `<span class="code-line ${hot.includes(start + i) ? "hot" : ""}"><span class="line-no">${start + i + (a.excerpt_start_line || 1)}</span><span class="code-text">${a.lines_html?.[start + i] ?? esc(l)}</span></span>`).join("")}</pre></div>`;
  }
  const focus = (a, needle, before = 1) =>
    Math.max(
      0,
      (a?.code?.split("\n").findIndex((s) => s.includes(needle)) ?? 0) - before,
    );
  function flatten(c, region = []) {
    return (c?.operations || []).flatMap((o) =>
      o.blocks
        ? o.blocks.flatMap((block, i) =>
            flatten({ operations: block }, [
              ...region,
              { id: o.id, name: o.name, iterations: o.iterations, branch: i },
            ]),
          )
        : [{ ...o, region }],
    );
  }
  function circuit(c, { active = -1, limit = 10, start = 0, label = "" } = {}) {
    if (!c)
      return `<p class="small">Circuit view unavailable for this intermediate form.</p>`;
    const ops = flatten(c);
    start = Math.max(0, Math.min(start, Math.max(0, ops.length - limit)));
    const shown = ops.slice(start, start + limit);
    const width = 1000,
      y0 = 104,
      dy = 77,
      height = Math.max(370, y0 + c.qubits.length * dy + 75);
    const gateStep = Math.min(88, 800 / Math.max(shown.length, 1));
    let body = c.qubits
      .map(
        (q, i) =>
          line(128, y0 + i * dy, 960, y0 + i * dy, "#9eb6cc", 2) +
          text(90, y0 + i * dy + 10, q.label, 30, "#142b45", "end"),
      )
      .join("");
    // Boundaries come from captured structured operations; loop bodies appear once.
    const regions = new Map();
    shown.forEach((o, j) =>
      o.region.forEach((r, depth) => {
        const key = `${r.id}:${r.branch}:${depth}`;
        const span = regions.get(key) || { ...r, depth, first: j, last: j };
        span.last = j;
        regions.set(key, span);
      }),
    );
    regions.forEach((r) => {
      const left = 180 + r.first * gateStep - 34,
        right = 180 + r.last * gateStep + 34;
      if (r.name === "for_loop")
        body +=
          `<rect class="region-bracket" x="${left}" y="58" width="${right - left}" height="${(c.qubits.length - 1) * dy + 96}" rx="9"/>` +
          text(
            (left + right) / 2,
            40,
            `repeat ${r.iterations}×`,
            27,
            "#2f70b8",
          );
      else
        body +=
          line(
            left,
            y0 + (c.qubits.length - 1) * dy + 44,
            right,
            y0 + (c.qubits.length - 1) * dy + 44,
            "#087d92",
            4,
          ) +
          text(
            (left + right) / 2,
            y0 + (c.qubits.length - 1) * dy + 73,
            "if 1",
            25,
            "#087d92",
          );
    });
    shown.forEach((o, j) => {
      const x = 180 + j * gateStep,
        hot = start + j === active,
        cls = hot ? "active-gate" : "gate";
      const ys = o.qubits.filter((q) => q >= 0).map((q) => y0 + q * dy);
      if (!ys.length) return;
      if (ys.length > 1)
        body += line(
          x,
          Math.min(...ys),
          x,
          Math.max(...ys),
          hot ? "#0756a8" : "#2f70b8",
          hot ? 6 : 4,
        );
      if (["cx", "cz"].includes(o.name)) {
        body += `<circle cx="${x}" cy="${ys[0]}" r="8" fill="#2f70b8"/>`;
        if (o.name === "cx")
          body +=
            `<circle class="${cls}" cx="${x}" cy="${ys.at(-1)}" r="23"/>` +
            line(x - 14, ys.at(-1), x + 14, ys.at(-1)) +
            line(x, ys.at(-1) - 14, x, ys.at(-1) + 14);
        else
          body += `<circle cx="${x}" cy="${ys.at(-1)}" r="${hot ? 11 : 8}" fill="#2f70b8"/>`;
      } else if (o.name === "swap") {
        ys.forEach((y) => {
          body +=
            line(
              x - 15,
              y - 15,
              x + 15,
              y + 15,
              hot ? "#0756a8" : "#2f70b8",
              5,
            ) +
            line(
              x - 15,
              y + 15,
              x + 15,
              y - 15,
              hot ? "#0756a8" : "#2f70b8",
              5,
            );
        });
      } else {
        const name =
          { measure: "M", reset: "|0⟩", r: "R", prx: "R" }[o.name] ||
          o.name.toUpperCase();
        ys.forEach((y) => {
          body +=
            `<rect class="${cls}" x="${x - 27}" y="${y - 26}" width="54" height="52" rx="7"/>` +
            text(x, y + 10, name, 29);
        });
      }
    });
    if (start + limit < ops.length) body += text(987, y0 + 10, "…", 34);
    body += text(
      500,
      height - 4,
      label ||
        (start
          ? `Captured operations ${start + 1}–${start + shown.length} of ${ops.length}`
          : "Actual program · loop body shown once"),
      24,
      "#5b7088",
    );
    return `<div class="circuit" data-op-count="${ops.length}">${svg(body, `0 0 ${width} ${height}`, "Actual captured circuit")}</div>`;
  }
  function topologyPositions() {
    const sites = [...data.device.sites].sort((a, b) => a.id - b.id),
      edges = new Set(
        data.device.edges.flatMap(([a, b]) => [`${a}:${b}`, `${b}:${a}`]),
      ),
      rows = [],
      points = new Map();
    sites.forEach((s) => {
      const row = rows.at(-1);
      if (row && edges.has(`${row.at(-1).id}:${s.id}`)) row.push(s);
      else rows.push([s]);
    });
    rows.forEach((row, y) => {
      let offset = 0;
      if (y) {
        const prior = rows[y - 1],
          i = row.findIndex((s) =>
            prior.some((p) => edges.has(`${s.id}:${p.id}`)),
          );
        if (i >= 0) {
          const p = prior.find((p) => edges.has(`${p.id}:${row[i].id}`));
          offset = points.get(p.id).x - i;
        }
      }
      row.forEach((s, x) => points.set(s.id, { x: x + offset, y }));
    });
    const all = [...points.values()],
      minX = Math.min(...all.map((p) => p.x)),
      maxX = Math.max(...all.map((p) => p.x)),
      maxY = Math.max(...all.map((p) => p.y));
    const scale = Math.min(640 / (maxX - minX), 435 / maxY);
    points.forEach((p) => {
      p.x = 400 + (p.x - (maxX + minX) / 2) * scale;
      p.y = 303 + (p.y - maxY / 2) * scale;
    });
    return points;
  }
  const points = topologyPositions();
  function topology(activeSites = []) {
    const active = new Set(activeSites);
    let body = text(400, 36, "IQM Emerald · 54 sites · 90 couplings", 29);
    body += data.device.edges
      .map(([a, b]) => {
        const p = points.get(a),
          q = points.get(b);
        return `<line class="topology-edge ${active.has(a) && active.has(b) ? "active" : ""}" x1="${p.x}" y1="${p.y}" x2="${q.x}" y2="${q.y}"/>`;
      })
      .join("");
    body += data.device.sites
      .map((s) => {
        const p = points.get(s.id);
        return `<g><circle class="topology-node ${active.has(s.id) ? "active" : ""}" cx="${p.x}" cy="${p.y}" r="21"/><text class="topology-label" x="${p.x}" y="${p.y + 8}">${s.id}</text></g>`;
      })
      .join("");
    body += text(
      400,
      585,
      "Exact coupling graph · topological layout",
      25,
      "#5b7088",
    );
    return `<div class="topology-svg">${svg(body, "0 0 800 610", "Emerald topology and highlighted captured operands")}</div>`;
  }
  function mappedScene(step, native = false) {
    const a = artifact(native ? "target-native-synthesis" : "place-and-route"),
      c = a?.circuit,
      ops = flatten(c);
    const selected = native ? [0, 1, 2, 3, 4, 5, 7] : [0, 1, 2, 3, 4, 5, 6];
    const idx = Math.min(selected[Math.max(0, step - 1)], ops.length - 1),
      op = step ? ops[idx] : null;
    const sites = op
      ? op.qubits
          .map((q) => c.qubits.find((w) => w.id === q)?.site)
          .filter(Number.isInteger)
      : variant().layout?.initial || [];
    const label = op
      ? `${op.name.toUpperCase()} · physical sites ${sites.join(" ↔ ")}`
      : "Logical qubits receive physical homes";
    return `<div class="architecture-grid"><div>${circuit(c, { active: step ? idx : -1, limit: 8, start: Math.max(0, idx - 3) })}<p class="operation-caption">${esc(label)}</p><p class="note">${native ? "Only the target’s R / CZ operations, measurement and reset remain." : "The circuit and topology highlight the same captured operands."}</p><p class="note">Initial placement</p><div class="mapping-legend">${(
      variant().layout?.initial || []
    )
      .slice(0, 3)
      .map((p, i) => `<span>q[${i}] → ${p}</span>`)
      .join(
        "",
      )}</div></div><div>${topology(sites)}</div></div>${sourceNote("Emerald connectivity · demo assumptions: reset and unrestricted classical control · execution on DDSIM")}`;
  }
  function pipeline(step) {
    return `<div class="pipeline">${["OpenQASM 3", "QC", "QCO", "Target", "Payload"].map((s, i) => `${i ? '<span class="arrow">→</span>' : ""}<span class="stage ${i === step ? "active" : ""}">${s}</span>`).join("")}</div>`;
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
    const groups = [
      ["Create session", "session_init"],
      ["Create job", "create_device_job"],
      ["Attach program", "set_programs"],
      ["Submit", "job_submit"],
      ["Wait / execute", "job_wait"],
      ["Read results", "get_results"],
    ];
    let body =
      text(130, 28, "Client", 27) +
      text(650, 28, "DDSIM", 27) +
      line(130, 50, 130, 340, "#b8ccdf", 2) +
      line(650, 50, 650, 340, "#b8ccdf", 2);
    groups.forEach(([label, key], i) => {
      const e = events.find((e) => e.operation.includes(key)),
        done = e && elapsed >= e.time_ms,
        inFlight =
          e && elapsed >= e.time_ms && elapsed < e.time_ms + e.duration_ms;
      const y = 76 + i * 48;
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
        label: "Actual capture client · Python → QDMI C ABI",
      };
    const srcLine = e?.source_line || 1;
    return code(a, {
      start: Math.max(0, srcLine - 2),
      count: 3,
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
    [0, 1, 2, 3].forEach((t) => {
      body += text(x(t), y0 + 39, t, 28, "#5b7088");
    });
    body +=
      `<path class="reference" d="M${x0} ${y(0)}H${x0 + w}"/>` +
      text(1280, y(0) + 8, "FCI", 27, "#078b9c", "start") +
      text(675, 525, "Imaginary time τ (Ha⁻¹)", 30) +
      text(105, 43, "Energy − FCI (mHa)", 30, "#142b45", "start");
    const count =
      step >= 3
        ? p.tau.length
        : Math.max(
            2,
            Math.round(
              p.tau.length * (step === 0 ? 0.05 : step === 1 ? 0.35 : 0.68),
            ),
          );
    curves.forEach((c, j) => {
      if (j === 1 && step < 3) return;
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
    if (step >= 3)
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
  function walkerFrame(step) {
    const frames = app.propagation?.curves?.[0]?.frames || [],
      f =
        frames[
          Math.min(
            frames.length - 1,
            Math.round((step / 3) * (frames.length - 1)),
          )
        ];
    if (!f) return "";
    const max = Math.max(...f.walkers.map((w) => w.weight)),
      x0 = 110,
      dx = 84,
      dy = 86;
    let body = f.walkers
      .map((w, i) => {
        const x = x0 + (i % 8) * dx,
          y = 55 + Math.floor(i / 8) * dy,
          r = 27 * Math.sqrt(w.weight / max);
        return `<g><circle cx="${x}" cy="${y}" r="${r}" fill="#2f70b8" opacity="${0.22 + (0.78 * w.weight) / max}"/><title>Walker ${w.id}, weight ${w.weight}, energy ${w.local_energy} Ha</title></g>`;
      })
      .join("");
    body += text(
      415,
      780,
      `τ = ${fmt(f.tau, 2)} Ha⁻¹ · ${f.walkers.length} recorded walkers`,
      31,
      "#2f70b8",
    );
    return svg(
      body,
      "0 0 840 810",
      "Actual walker weights; fixed grid is a schematic arrangement",
    );
  }
  const slides = [];
  function add(section, title, builds, render, notes = "") {
    slides.push({ section, title, builds, render, notes });
  }
  add(
    "Munich Quantum Software Forum · 14–15 October 2026",
    "System Software for Quantum Computing",
    0,
    () =>
      `<div class="title-layout"><div><div class="talk-date">MQSF 2026 · Munich</div><h1>System Software<br>for Quantum Computing:<br><span class="blue">From the Metal<br>to the User</span></h1><p class="byline">Lukas Burgholzer</p><p class="affiliation">CTO &amp; Co-founder · MQSC<br>Senior Researcher · Technical University of Munich</p></div><div>${logo("q", "title-mark")}</div></div>`,
    "Opening: hardware is moving fast. This talk follows the software work that makes a quantum program usable inside a real computing workflow.",
  );
  add(
    "01 · The missing middle",
    "Hardware scales. Software breaks.",
    1,
    (s) =>
      heading(
        "Hardware scales.<br><span class='blue'>Software breaks.</span>",
      ) +
      `<div class="scene" style="height:550px">${viz.ecosystem(s === 0 ? 0 : 1)}</div>`,
    "Build researchers and HPC, then hardware, then the many bespoke connections. Name the integration burden, not a hypothetical failure of every product.",
  );
  add(
    "01 · The missing middle",
    "Connect once. Compose the workflow.",
    3,
    (s) =>
      heading(
        "Give the workflow a <span class='blue'>shared foundation.</span>",
      ) +
      `<div class="scene">${viz.ecosystem(s < 2 ? 2 : 3)}</div>` +
      reveal(
        s,
        3,
        sourceNote("Open interfaces · vendor-neutral infrastructure · mq.sc"),
      ),
    "The same scene resolves into a shared stack. The software middle is the subject of the keynote.",
  );
  add(
    "01 · The missing middle",
    "Open the software stack.",
    3,
    (s) =>
      heading(
        "Open the software stack.",
        "A compiler, a device interface, and a place in the computing system.",
      ) + `<div class="scene" style="height:550px">${viz.stack(s)}</div>`,
    "Reveal device access, runtime, then compiler. Use the stack to orient the audience; return to it at the end.",
  );
  add(
    "02 · From circuits to programs",
    "A circuit is only part of a program.",
    3,
    (s) =>
      heading(
        "A circuit is only <span class='blue'>part of a program.</span>",
      ) +
      `<div class="split"><div>${circuit(artifact("qc")?.circuit, { limit: 8, active: [-1, 6, 2, -1][s] })}</div><div class="body-copy">${reveal(s, 0, '<p class="big">Quantum operations</p>')}${reveal(s, 1, '<p class="big blue">Measurement feedback</p>')}${reveal(s, 2, '<p class="big blue">Loops and reset</p>')}${reveal(s, 3, '<div class="rule"></div><p>Keep the program’s meaning<br>through every compiler stage.</p>')}</div></div>`,
    "Bridge from last year’s Beyond Circuits. Error correction and adaptive algorithms are programs, not long static gate lists.",
  );
  add(
    "02 · One recognizable example",
    "Measure. Correct. Repeat.",
    3,
    (s) =>
      heading(
        "Measure. Correct. Repeat.",
        "Three qubits. Two rounds. One measurement-controlled correction.",
      ) +
      `<div class="split wide-left"><div>${code(artifact("source"), { count: 12, hot: s === 1 ? [5, 10] : s === 2 ? [8] : s === 3 ? [9] : [], title: "Actual OpenQASM input" })}</div><div>${circuit(artifact("source")?.circuit, { limit: 9, active: [-1, 2, 5, 6][s] })}<p class="statement">${["Prepare two data qubits.", "Repeat the parity check twice.", "Measure parity with an ancilla.", "Correct the data when parity is odd."][s]}</p></div></div>`,
    "Explain data qubits and the parity ancilla before syntax. This is a toy feedback program, not a full error-correction protocol. Highlight the loop, measurement, then correction.",
  );
  add(
    "02 · One recognizable example",
    "Follow the same three wires.",
    5,
    (s) =>
      heading("Follow the same three wires.") +
      `<div style="margin-top:90px">${circuit(artifact("source")?.circuit, { limit: 10, active: s < 5 ? s + 2 : -1 })}</div><p class="statement blue">${["Reset the ancilla.", "Write the first data qubit’s parity.", "Combine the second data qubit.", "Measure the ancilla.", "Apply the conditional correction.", "Keep the loop around the whole operation."][s]}</p>` +
      sourceNote(
        "Circuit drawn from the actual imported program · loop bodies are shown once",
      ),
    "Use the clicker to walk the circuit before introducing MLIR. Repetition and feedback are visible without reading code.",
  );
  add(
    "03 · The compiler",
    "Change the representation. Keep the structure.",
    4,
    (s) =>
      heading(
        "Change the representation.<br><span class='blue'>Keep the structure.</span>",
      ) +
      pipeline(s) +
      `<div class="split"><div class="body-copy"><p class="big">${["Read a quantum program.", "Make side effects explicit.", "Track quantum values.", "Fit the selected device.", "Emit an executable payload."][s]}</p><div class="rule"></div><p>${["OpenQASM 3 and existing frontends enter the same compiler.", "QC expresses operations on quantum references.", "QCO gives optimizations explicit quantum data flow.", "Placement, routing and native-gate synthesis share device information.", "OpenQASM 3 and Adaptive QIR retain supported classical control."][s]}</p></div><div>${logo("mqt")}<p class="statement">Built on MLIR</p><p class="note">Classical compiler infrastructure.<br>Quantum semantics.</p></div></div>`,
    "This pipeline is the audience’s map. Explain why multiple representations exist with one sentence each; do not teach all MLIR syntax.",
  );
  add(
    "03 · The compiler",
    "The loop survives the language change.",
    2,
    (s) => {
      const a = artifact(s === 0 ? "source" : s === 1 ? "qc" : "qco"),
        start = s ? focus(a, "scf.for", 0) : 4;
      return (
        heading("The loop survives the language change.") +
        pipeline(s) +
        `<div class="split code-focus"><div>${code(a, { start, count: s ? 5 : 8, title: s === 0 ? "OpenQASM 3" : s === 1 ? "Actual QC intermediate representation" : "Actual value-based QCO", compact: true })}</div><div><p class="big blue">${["for", "scf.for", "scf.for"][s]}</p><p class="statement">One loop.<br>Quantum operations.<br>Classical feedback.</p><p class="note">Exact excerpt from this compilation.<br>Full artifacts are in the offline ZIP.</p></div></div>`
      );
    },
    "Two clicks transform the same program through source, QC, QCO. Read just the loop and one quantum operation. The rest supplies evidence, not a reading assignment.",
  );
  add(
    "03 · Compilation meets the machine",
    "Give every qubit a physical home.",
    7,
    (s) => heading("Give every qubit a physical home.") + mappedScene(s, false),
    "The left circuit and right Emerald topology come from the same captured mapping. Click through real gate operands. The grid is a topological layout, not chip coordinates. Do not claim a routing SWAP when none was needed.",
  );
  add(
    "03 · Compilation meets the machine",
    "Speak the device’s gate language.",
    7,
    (s) =>
      heading(
        "Speak the device’s gate language.",
        "Synthesize to R and CZ while preserving measurement and feedback.",
      ) + mappedScene(s, true),
    "Now the circuit is native. Highlight real R/CZ operations and their physical sites together. State that reset and unrestricted control are presentation assumptions; the payload runs on DDSIM.",
  );
  add(
    "03 · Control is a capability",
    "Keep the loop—or expand the bounded repetition.",
    2,
    (s) =>
      heading("Keep the loop—or expand the bounded repetition.") +
      `<div class="split"><div><span class="tag">Preserved structure</span>${circuit(artifact("optimized")?.circuit, { limit: 9 })}<p class="statement">One loop body</p></div>${reveal(s, 1, `<div><span class="tag">Bounded loop unrolled</span>${circuit(artifact("unrolled", true)?.circuit, { limit: 15 })}<p class="statement">Two copies of the body</p></div>`)}</div>${reveal(s, 2, '<p class="build-callout">Measurement-dependent corrections remain conditional in both outputs.</p>')}`,
    "Unrolling a bounded repetition is different from removing measurement feedback. Both current outputs require adaptive execution.",
  );
  add(
    "03 · Portable output",
    "One compiled program. Two payload formats.",
    1,
    (s) => {
      const a = artifact(s ? "qir-adaptive" : "openqasm3"),
        needle = s ? "br i1" : "if (";
      return (
        heading(
          "One compiled program.<span class='blue'> Two payload formats.</span>",
        ) +
        `<div class="split code-focus"><div>${code(a, { start: focus(a, needle, 1), count: s ? 7 : 9, title: s ? "Actual Adaptive QIR" : "Actual native OpenQASM 3", compact: true })}</div><div><p class="huge blue">${s ? "QIR" : "QASM 3"}</p><p class="statement">${s ? "Classical LLVM control.<br>Quantum runtime calls." : "A structured program.<br>Device-native operations."}</p><p class="note">Switching formats is one click.<br>The payload itself is captured compiler output.</p></div></div>`
      );
    },
    "Show the exact branch in each emitted format. No dropdown: the next press changes format. Qiskit is available as a backup reference if needed.",
  );
  add(
    "04 · The device conversation",
    "The compiler needs answers from the device.",
    3,
    (s) =>
      heading("The compiler needs answers from the device.") +
      `<div class="split"><div>${topology(s >= 1 ? variant().layout?.initial || [] : [])}</div><div class="body-copy">${reveal(s, 0, "<p>Which qubits can interact?</p>")}${reveal(s, 1, "<p>Which operations are native?</p>")}${reveal(s, 2, "<p>Which program formats and<br>control-flow features are supported?</p>")}${reveal(s, 3, `<div class="rule"></div>${logo("qdmi")}<p class="statement">One interface for those answers.</p>`)}</div></div>`,
    "Move from compiler internals to hardware information. These are the same questions raised in the AWS tutorial; QDMI is the conversation, not just a submission API.",
  );
  add(
    "04 · The device conversation",
    "Discover. Submit. Retrieve.",
    3,
    (s) =>
      heading("Discover. Submit. Retrieve.") +
      `<div class="split"><div>${logo("qdmi")}<p class="lead" style="margin-top:40px">One open interface.<br>Hardware, cloud services,<br>and simulation engines.</p>${reveal(s, 3, `<div style="margin-top:40px">${qr("qdmi", "mq.sc/#qdmi-title", "Quantum Device Management Interface")}</div>`)}</div><div>${sequence(run, s === 0 ? 0 : s === 1 ? run.submitted_ms || 0 : s === 2 ? run.completed_ms || 0 : Infinity)}</div></div>`,
    "Three verbs are enough to read the next demonstration. Reveal lifecycle groups; these are actual calls observed at the DDSIM device API boundary.",
  );
  add(
    "04 · Execute a richer program",
    "An inexact phase produces a distribution.",
    2,
    (s) =>
      heading(
        "An inexact phase produces a distribution.",
        "Iterative QPE · four logical qubits · five mapped sites · eight phase bits · phase = 1/3",
      ) +
      `<div class="split"><div><div class="metric-row">${metric("1/3", "true phase")}${metric("256", "readout bins")}</div><p class="statement">1/3 falls between<br><span class="blue">85/256 and 86/256.</span></p>${reveal(s, 1, '<p class="lead" style="margin-top:30px">One query qubit is reset and reused.<br>Three qubits form the eigenstate register.</p>')}</div><div>${histogram(run, s < 2 ? 0 : Infinity)}</div></div>`,
    "The compiler teaching example and execution workload are intentionally different. Explain phase estimation before showing a runtime trace. The histogram is the actual 2048-shot run.",
  );
  add(
    "04 · Execute a richer program",
    "First, the run at its recorded speed.",
    1,
    (s) =>
      heading(
        "First, the run at its recorded speed.",
        "One click starts the captured run at 1× time.",
      ) + executionScene(s ? totalTime() : 0, "Recorded time · 1×"),
    "Press once. At recorded speed this is too fast to explain, so the next slide replays the same timeline slowly. This is recorded ideal DDSIM execution, not a remote hardware run.",
  );
  add(
    "04 · Execute a richer program",
    "Now slow down the same clock.",
    1,
    (s) =>
      heading(
        "Now slow down the same clock.",
        "Code, calls and shot completions share one measured timeline.",
      ) +
      executionScene(
        s ? totalTime() : 0,
        `Recorded time · ${fmt(Math.max(1, 18000 / totalTime()), 0)}× slower`,
      ),
    "Press once for an 18-second replay. Code follows the real call, wait spans execution, and counts rise only at actual shot timestamps. Press Next during playback to finish this build; press again to advance. Back rewinds deterministically.",
  );
  add(
    "05 · A reason to connect the stack",
    "A molecule is the question—not a circuit.",
    3,
    (s) =>
      heading(
        "A molecule is the question—<span class='blue'>not a circuit.</span>",
        "Quantum-assisted auxiliary-field quantum Monte Carlo (AFQMC)",
      ) +
      `<div class="scene" style="height:540px">${viz.molecule(s)}</div>` +
      sourceNote(
        "Demonstration: H₂ at 0.75 Å · STO-3G · two electrons in four spin orbitals · ideal local simulation",
      ),
    "Change scale: the scientist wants a molecular energy. Explain that this deliberately small H2 instance is a complete, checkable integration demonstration, not a quantum-advantage claim.",
  );
  add(
    "05 · Quantum work becomes reusable data",
    "Prepare. Randomize. Measure.",
    3,
    (s) =>
      heading(
        "Prepare. Randomize. Measure.",
        "Matchgate shadows turn repeated quantum measurements into reusable classical data.",
      ) +
      `<div class="scene" style="height:550px">${viz.shadows(s, app.snapshots?.[0]?.shots?.[0] || "measured bits")}</div>`,
    "Follow trial-state preparation into random Gaussian rotations, measurement, and a stored basis/outcome pair. The shown sample comes from the captured batch. The full method follows the public Braket AFQMC example with a verified sign-convention adaptation.",
  );
  add(
    "05 · A real multi-program job",
    "512 programs. One QDMI job.",
    2,
    (s) =>
      heading(
        `${fmt(app.workload?.snapshots || 0, 0)} programs.<span class='blue'> One QDMI job.</span>`,
      ) +
      `<div class="split"><div>${code(
        {
          code: app.source || "",
          lines_html: app.source_lines_html,
          language: "python",
          label: "Actual PennyLane trial-state circuit",
        },
        {
          start: Math.max(
            0,
            (app.source || "")
              .split("\n")
              .findIndex((l) => l.includes("qml.Hadamard")),
          ),
          count: 9,
          compact: true,
        },
      )}</div><div><div class="metric-row">${metric(fmt(app.workload?.snapshots, 0), "programs")}${metric(fmt(app.workload?.shots_per_snapshot, 0), "shots / program")}</div>${reveal(s, 1, `<div class="metric-row">${metric(fmt(app.workload?.total_shots, 0), "recorded outcomes")}</div>`)}${reveal(s, 2, `<p class="statement blue">${fmt(app.workload?.batch_duration_ms, 1)} ms</p><p class="note">Observed whole PennyLane batch call.<br>Indexed payloads, shots and counts verified.</p>`)}</div></div>`,
    "The updated main branch supports a native multi-program job. This number is observed job count, not a presentation grouping of separate jobs. State the whole-call timing scope.",
  );
  add(
    "05 · Put the measurements to work",
    "Quantum data guides classical walkers.",
    3,
    (s) =>
      heading("Quantum data guides classical walkers.") +
      `<div class="split wide-left"><div class="scene" style="height:560px">${walkerFrame(s)}</div><div><p class="eyebrow">Classical propagation</p><p class="big">An ensemble of<br>Slater determinants</p><p class="statement">Propagate → evaluate overlaps<br>→ reweight → estimate energy</p><p class="note">Circle area reflects recorded walker weight.<br>The grid is a schematic arrangement.</p><div class="metric-row">${metric(app.propagation?.walkers || 0, "walkers")}${metric(app.propagation?.steps || 0, "steps")}</div></div></div>`,
    "A walker is a numerical representation of electronic state, not an electron trajectory. The plotted sizes are actual saved weights. Trial overlaps from shadows enter the unchanged upstream phaseless propagation algorithm.",
  );
  add(
    "05 · Close the loop with evidence",
    "A complete small chemistry calculation.",
    3,
    (s) =>
      heading(
        "A complete small chemistry calculation.",
        "Actual phaseless AFQMC trajectories, checked against full configuration interaction.",
      ) +
      energyChart(s) +
      `<p class="note">Band: ±1 walker-only standard error. Finite-shadow error, time-step error and phaseless bias are not included.</p>` +
      sourceNote(
        "H₂ / STO-3G · independent six-determinant Hamiltonian agrees with PySCF FCI · no quantum-advantage claim",
      ),
    "Grow the recorded trajectory, then reveal the Hartree-Fock-trial comparison. Explain the band precisely. The last point is a final-time snapshot, not an independent tail-averaged estimate. Small-model overlap contraction is exact for the reconstructed six coefficients, not a scalable replacement for AFQMC.",
  );
  add(
    "05 · The application crosses both worlds",
    "The useful unit is the whole workflow.",
    3,
    (s) =>
      heading(
        "The useful unit is <span class='blue'>the whole workflow.</span>",
      ) +
      `<div class="scene" style="height:520px">${viz.walkers(s)}</div>` +
      reveal(
        s,
        3,
        `<div class="metric-row" style="justify-content:center;margin-top:0">${metric(fmt(app.workload?.batch_duration_ms, 1) + " ms", "quantum batch on DDSIM", true)}${metric(fmt((app.propagation?.duration_ms || 0) / 1000, 2) + " s", "classical propagation", true)}</div>`,
      ),
    "Pull back from the result to the application architecture. Timings are measured scopes from this local demo and are not a speedup claim. Native QDMI batching lets the program collection travel through the same interface.",
  );
  add(
    "06 · From the metal to the user",
    "Fit quantum work into existing infrastructure.",
    3,
    (s) =>
      heading("Fit quantum work into existing infrastructure.") +
      `<div class="scene">${viz.stack(s)}</div>` +
      sourceNote(
        "Deployment context: Slurm / cloud orchestration · the chemistry capture shown here ran locally on DDSIM",
      ),
    "Use the familiar stack again, now from the application’s viewpoint. The AWS tutorial shows Slurm/cloud deployment; distinguish that architecture from the local measured chemistry run. QPUs should behave as accelerator resources inside workflows people already use.",
  );
  add(
    "06 · Evidence and ecosystem",
    "Coverage. Circuit quality. Compilation cost.",
    2,
    (s) =>
      heading("Coverage. Circuit quality. Compilation cost.") +
      `<div class="takeaways"><div><h3>Coverage</h3><p>Every compiler.<br>Every circuit in the<br>declared benchmark suite.</p></div>${reveal(s, 1, "<div><h3>Circuit quality</h3><p>Compare native two-qubit<br>gates on matched targets.</p></div>")}${reveal(s, 2, "<div><h3>Compilation cost</h3><p>Report runtime together<br>with output quality.</p></div>")}</div><div class="rule" style="margin-top:65px"></div><p class="statement">Full Benchpress comparison: final measurements pending.</p><p class="note">This rehearsal slide makes no coverage or performance-superiority claim.</p>`,
    "This slot is reserved for the final full-suite data. Do not present speculative numbers or reuse the medium subset as a full-suite result. Replace this slide with the sourced graph before the event.",
  );
  add(
    "06 · Evidence and ecosystem",
    "Built in the open. Built together.",
    2,
    (s) =>
      heading("Built in the open.<span class='blue'> Built together.</span>") +
      `<div class="logos">${logo("mqt")}${logo("qdmi")}${logo("mqss")}</div>${reveal(s, 1, `<div class="logos">${logo("tum-cda")}${logo("mqv")}</div><p class="statement">Research, open interfaces, and engineering—across the ecosystem.</p>`)}${reveal(s, 2, `<div style="margin-top:55px">${qr("core", "mqt.readthedocs.io", "Explore the compiler, interfaces, examples and documentation")}</div>`)}`,
    "Credit the TUM Chair for Design Automation, MQT contributors, QDMI and MQSS partners, and Munich Quantum Valley. MQSC is the primary speaking affiliation; research and community contributions are visible and explicit.",
  );
  add(
    "From the Metal to the User",
    "Hardware scales. Put the software to work.",
    2,
    (s) =>
      heading(
        "Hardware scales.<br><span class='blue'>Put the software to work.</span>",
      ) +
      `<div class="takeaways"><div><h3>Keep the structure.</h3><p>Compile quantum programs<br>with classical control.</p></div>${reveal(s, 1, "<div><h3>Connect the device.</h3><p>Discover, submit, retrieve<br>through open interfaces.</p></div>")}${reveal(s, 2, "<div><h3>Run the workflow.</h3><p>Turn quantum work into<br>useful application data.</p></div>")}</div><div class="final-resources">${qr("company", "mq.sc", "System software for quantum computing")}${qr("core", "mqt.readthedocs.io", "Open software. Real examples. Join us.")}</div>`,
    "Close on the three demonstrated outcomes, not a product menu. Invite the audience to bring programs and devices. Keep the QR codes visible for questions.",
  );

  function resize() {
    document.documentElement.style.setProperty(
      "--scale",
      Math.min(innerWidth / 1920, innerHeight / 1080),
    );
  }
  let transition = null;
  function render() {
    state.token++;
    state.animation = null;
    transition?.skipTransition();
    if (
      document.startViewTransition &&
      state.slide !== 16 &&
      state.slide !== 17 &&
      !matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
      transition = document.startViewTransition(draw);
      // Rapid clicker presses may intentionally cancel the visual transition.
      transition.ready.catch(() => {});
      return;
    }
    draw();
  }
  function draw() {
    const slide = slides[state.slide];
    $("section-label").textContent = slide.section;
    $("slide").innerHTML = slide.render(state.step);
    $("slide").dataset.slide = String(state.slide + 1);
    $("slide").dataset.step = String(state.step);
    $("slide-number").innerHTML =
      `${String(state.slide + 1).padStart(2, "0")} / ${slides.length}<span class="build-dots" aria-label="Build ${state.step + 1} of ${slide.builds + 1}">${Array.from({ length: slide.builds + 1 }, (_, i) => (i === state.step ? "<b>•</b>" : "•")).join("")}</span>`;
    $("build-progress").style.width =
      `${((state.slide + (state.step + 1) / (slide.builds + 1)) / slides.length) * 100}%`;
    $("speaker-notes").textContent =
      `${state.slide + 1}. ${slide.title} — ${slide.notes}`;
    history.replaceState(null, "", `#${state.slide + 1}.${state.step}`);
  }
  function animateRuntime(slow) {
    const token = state.token,
      begin = performance.now(),
      duration = slow ? 18000 : totalTime();
    const end = () => {
      if (token !== state.token) return;
      state.animation = null;
      $("slide").innerHTML = slides[state.slide].render(state.step);
    };
    state.animation = { end };
    function frame(now) {
      if (token !== state.token || !state.animation) return;
      const elapsed = Math.min(
        totalTime(),
        ((now - begin) / duration) * totalTime(),
      );
      $("slide").innerHTML =
        heading(
          slow
            ? "Now slow down the same clock."
            : "First, the run at its recorded speed.",
          slow
            ? "Code, calls and shot completions share one measured timeline."
            : "Recorded ideal DDSIM execution at 1× time.",
        ) +
        executionScene(
          elapsed,
          slow
            ? `${fmt(duration / totalTime(), 0)}× slower · same measured timeline`
            : "Recorded time · 1×",
        );
      if (elapsed < totalTime()) requestAnimationFrame(frame);
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
      state.animation.end();
      return;
    }
    const current = slides[state.slide];
    if (state.step < current.builds) {
      state.step++;
      render();
      if (state.slide === 16 || state.slide === 17) {
        animateRuntime(state.slide === 17);
      }
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
      title: s.title,
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
  };
  $("mqsc-logo").src = assets.mqsc;
  resize();
  const hash = location.hash.match(/^#(\d+)(?:\.(\d+))?$/);
  go(hash ? Number(hash[1]) - 1 : 0, hash ? Number(hash[2] || 0) : 0);
  $("deck").focus();
})();
