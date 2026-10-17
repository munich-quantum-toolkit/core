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
window.MQSF_CIRCUIT = (() => {
  const esc = (s) =>
    String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll('"', "&quot;");
  function flatten(c, region = []) {
    return (c?.operations || []).flatMap((o) =>
      o.blocks
        ? o.blocks.flatMap((block, branch) =>
            flatten({ operations: block }, [
              ...region,
              { ...o, blocks: undefined, branch },
            ]),
          )
        : [{ ...o, region }],
    );
  }
  function schedule(c) {
    const free = new Map(c.qubits.map((q) => [q.id, 0]));
    const rows = new Map(c.qubits.map((q, i) => [q.id, i]));
    const bits = new Map();
    let boundary = "",
      floor = 0;
    return flatten(c).map((o, index) => {
      const nextBoundary = o.region.map((r) => `${r.id}:${r.branch}`).join("/");
      // Never move an operation across a loop or branch boundary.
      if (nextBoundary !== boundary) floor = Math.max(...free.values(), floor);
      boundary = nextBoundary;
      const used = (
        o.qubits.length ? o.qubits : c.qubits.map((q) => q.id)
      ).filter((q) => rows.has(q));
      const lo = Math.min(...used.map((q) => rows.get(q))),
        hi = Math.max(...used.map((q) => rows.get(q)));
      const span = c.qubits
        .filter((_, row) => row >= lo && row <= hi)
        .map((q) => q.id);
      const dependencies = o.region.flatMap((r) => r.condition_bits || []);
      const column = Math.max(
        floor,
        ...span.map((q) => free.get(q)),
        ...dependencies.map((b) => bits.get(b) || 0),
      );
      span.forEach((q) => free.set(q, column + (o.name === "store" ? 2 : 1)));
      (o.clbits || []).forEach((b) => bits.set(b, column + 1));
      return { ...o, index, column };
    });
  }
  function gateLabel(name) {
    const rotation = /^r([xyz]{1,2})$/.exec(name);
    if (rotation)
      return `R<tspan baseline-shift="sub" font-size="70%">${rotation[1]}</tspan>`;
    return esc(
      { reset: "|0⟩", prx: "R", gpi: "GPI", gpi2: "GPI₂" }[name] ||
        name.toUpperCase(),
    );
  }
  function render(
    c,
    {
      active = -1,
      start = 0,
      limit = 16,
      label = "",
      key = "circuit",
      layout = null,
      camera = null,
      full = false,
      reveal = Infinity,
      barrier = null,
      hotRegion = null,
      fold = null,
    } = {},
  ) {
    if (!c)
      return '<p class="note">No circuit representation for this compiler stage.</p>';
    const all = schedule(c),
      maximum = Math.max(0, ...all.map((o) => o.column));
    const highlighted = new Set(Array.isArray(active) ? active : [active]);
    const anchor =
      camera ??
      all[Array.isArray(active) ? active[0] : active >= 0 ? active : start]
        ?.column ??
      0;
    const firstColumn = full
      ? 0
      : Math.max(
          0,
          Math.min(
            anchor - (camera !== null || active >= 0 ? 5 : 0),
            Math.max(0, maximum - limit + 1),
          ),
        );
    const windowColumn = Math.floor(firstColumn);
    const operations = all.filter(
      (o) =>
        o.column >= windowColumn &&
        o.column < windowColumn + (full ? maximum + 1 : limit),
    );
    const columns = full
      ? maximum + 1
      : Math.min(limit, maximum - windowColumn + 1);
    const structured = all.some((o) => o.region.length),
      nested = Math.max(0, ...all.map((o) => o.region.length));
    // Fold only straight-line circuits; structured regions keep their visible scope.
    const bandColumns =
        full && fold && !structured ? Math.min(columns, fold) : columns,
      bands = Math.ceil(columns / bandColumns),
      band = (o) => Math.floor((o.column - windowColumn) / bandColumns);
    // Gate bodies never exceed their column; a full circuit expands its SVG rather than overlapping.
    const pitch = full ? 76 : Math.max(76, Math.min(102, 900 / columns));
    const left = structured ? (layout ? 250 : 130) : layout ? 140 : 85,
      first = left + 40;
    const width = first + (bandColumns - 1) * pitch + 45,
      top = structured ? 100 + nested * 20 : 35,
      dy = structured ? 72 : 60;
    const bandHeight = Math.max(
        structured ? 280 : 240,
        top + (c.qubits.length - 1) * dy + (structured ? 120 : 40),
      ),
      height = bands * bandHeight;
    const row = new Map(c.qubits.map((q, i) => [q.id, i])),
      y = (q, b = 0) => top + row.get(q) * dy + b * bandHeight;
    const x = (o) => first + ((o.column - windowColumn) % bandColumns) * pitch;
    const txt = (xx, yy, t, size = 26, color = "#142b45", anchor = "middle") =>
      `<text x="${xx}" y="${yy}" font-size="${size}" fill="${color}" text-anchor="${anchor}">${esc(t)}</text>`;
    const ln = (x1, y1, x2, y2, color = "#2f70b8", w = 2) =>
      `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"/>`;
    const regions = new Map();
    operations.forEach((o) =>
      o.region.forEach((r, depth) => {
        const entry = regions.get(r.id) || {
          first: x(o),
          last: x(o),
          r,
          depth,
          qubits: new Set(),
        };
        entry.last = x(o) + (o.name === "store" ? 60 : 0);
        (o.qubits.length ? o.qubits : r.qubits || []).forEach((q) =>
          entry.qubits.add(q),
        );
        regions.set(r.id, entry);
      }),
    );
    let body = "";
    const captions = [];
    regions.forEach(({ first: a, last: b, r, depth, qubits }) => {
      const loop = ["for_loop", "while_loop"].includes(r.name);
      if (!loop && r.name !== "if_else") return;
      const ys = [...qubits].filter((q) => row.has(q)).map((q) => y(q));
      if (!ys.length) return;
      const yy = loop ? top - 66 - depth * 18 : Math.min(...ys) - 30;
      const bottom = loop
        ? top + (c.qubits.length - 1) * dy + 36
        : Math.max(...ys) + 30;
      const pad = loop ? 32 : 28,
        selected = hotRegion === r.name;
      // shortcut: condition labels cover these captured bit guards; general expressions need a formatter.
      const caption =
        r.name === "for_loop"
          ? `${r.iterations ?? "n"} rounds`
          : r.name === "while_loop"
            ? "Repeat Until Success"
            : `if ${r.condition?.includes("BIT_XOR") ? "¬" : ""}${(r.condition_bits || []).map((bit) => c.clbits?.find((b) => b.id === bit)?.label || `b${bit}`).join(", ")}`;
      const labelX = a - pad + 12,
        labelWidth = caption.length * 14;
      let labelY = yy - 10;
      while (
        captions.some(
          (previous) =>
            labelX < previous.right &&
            labelX + labelWidth > previous.left &&
            Math.abs(labelY - previous.y) < 28,
        )
      )
        labelY -= 28;
      captions.push({ left: labelX, right: labelX + labelWidth, y: labelY });
      body += `<g data-morph="${key}-region-${r.id}" data-control-flow="${r.name}"><rect x="${a - pad}" y="${yy}" width="${b - a + 2 * pad}" height="${bottom - yy}" rx="10" fill="${loop ? "#eaf2fa" : "#e9f5f3"}" fill-opacity="${selected ? ".9" : ".45"}" stroke="${selected ? "#086f83" : loop ? "#87acd0" : "#55a194"}" stroke-width="${selected ? 4 : 2}" ${loop ? "" : 'stroke-dasharray="5 4"'}/>${txt(labelX, labelY, caption, 23, loop ? "#2f70b8" : "#087d92", "start")}${loop ? `<path d="M${b - 8} ${bottom + 11}H${a + 8}m10 -8l-10 8 10 8" fill="none" stroke="#6992bc" stroke-width="2"/>` : ""}</g>`;
    });
    body += Array.from({ length: bands }, (_, b) =>
      c.qubits
        .map((q) => {
          const logical =
              layout && q.site != null ? layout.indexOf(q.site) : row.get(q.id),
            physical = q.site ?? layout?.[logical] ?? q.id;
          const wireLabel =
            layout && logical >= 0
              ? `<text x="${left - 20}" y="${y(q.id, b) + 8}" font-size="24" fill="#526d88" text-anchor="end" font-family="STIX Two Math,serif">q<tspan baseline-shift="sub" font-size="75%">${logical}</tspan><tspan baseline-shift="baseline"> ↦ ${physical}</tspan></text>`
              : txt(
                  left - 18,
                  y(q.id, b) + 8,
                  String(q.label).replace(/^\$(.*)\$$/, "$1"),
                  24,
                  "#526d88",
                  "end",
                );
          const right =
            first +
            (Math.min(bandColumns, columns - b * bandColumns) - 1) * pitch +
            25;
          return `<g data-morph="${key}-wire-${q.id}-${b}">${ln(left, y(q.id, b), right, y(q.id, b), "#b6c9dc", 2)}${wireLabel}${b < bands - 1 ? txt(right + 13, y(q.id, b) + 8, "›", 29, "#526d88") : ""}${b ? txt(left - 4, y(q.id, b) + 8, "›", 29, "#526d88") : ""}</g>`;
        })
        .join(""),
    ).join("");
    if (barrier !== null) {
      const next = all[barrier];
      if (next) {
        const bx = x(next) - pitch / 2;
        body += `<path d="M${bx} ${top - 25}V${top + (c.qubits.length - 1) * dy + 25}" stroke="#718ba6" stroke-width="2" stroke-dasharray="5 5"/>${txt((left + bx) / 2, top - 32, "Trial State", 23, "#58718b")}${txt((bx + width) / 2, top - 32, "Classical Shadow", 23, "#58718b")}`;
      }
    }
    const measured = new Map(),
      feedback = new Set();
    operations.forEach((o) => {
      const xx = x(o),
        ys = o.qubits.filter((q) => row.has(q)).map((q) => y(q, band(o))),
        hot = highlighted.has(o.index);
      if (!ys.length) {
        if (o.name === "store" && o.region.some((r) => r.branch === 1)) {
          const bit = c.clbits?.find((b) => b.id === o.classical_writes?.[0]);
          const value = /^Value\(([01]), Bool\(\)\)$/.exec(
            o.parameters?.[1] || "",
          )?.[1];
          const label =
            value === undefined
              ? "Classical Store"
              : `else: ${bit?.label || "bit"} ← ${value}`;
          body += `<g data-morph="${key}-op-${o.id}" data-operation="store" data-column="${o.column}" transform="translate(${xx} 0)"><rect x="-35" y="${top + (c.qubits.length - 1) * dy - 15}" width="125" height="40" rx="6" fill="#e8f5f0" stroke="#55a194"/>${txt(37, top + (c.qubits.length - 1) * dy + 10, label, 18, "#087d92")}</g>`;
        }
        return;
      }
      const color = hot ? "#0756a8" : "#2f70b8",
        fill = hot ? "#cee6fc" : "white";
      const dot = (yy, i = 0) =>
        `<circle cx="0" cy="${yy}" r="7" fill="${o.control_state === undefined || o.control_state & (1 << i) ? color : "white"}" stroke="${color}" stroke-width="2"/>`;
      const box = (yy, name) =>
        `<rect x="-27" y="${yy - 25}" width="54" height="50" rx="5" fill="${fill}" stroke="${color}" stroke-width="${hot ? 4 : 2.3}"/><text x="0" y="${yy + 9}" text-anchor="middle" font-size="${name.length > 5 ? 19 : 27}" fill="${color}" font-family="STIX Two Math,serif">${gateLabel(name)}</text>`;
      const controlCount =
        o.controls ??
        o.num_controls ??
        (o.name.startsWith("cc")
          ? 2
          : o.name.startsWith("c") && o.name !== "ctrl"
            ? 1
            : 0);
      let gate = "";
      if (
        (controlCount > 0 && o.base_gate) ||
        [
          "cx",
          "cz",
          "ccx",
          "ccz",
          "cp",
          "crx",
          "cry",
          "crz",
          "ch",
          "csx",
          "cy",
          "cu",
        ].includes(o.name)
      ) {
        gate += ln(0, Math.min(...ys), 0, Math.max(...ys), color, 2.5);
        const controls = ys.slice(0, controlCount || 1),
          target = ys.at(-1);
        gate += controls.map((yy, i) => dot(yy, i)).join("");
        const base = o.base_gate || o.name.replace(/^c+/, "");
        if (base === "x")
          gate += `<circle cx="0" cy="${target}" r="21" fill="${fill}" stroke="${color}" stroke-width="2.5"/>${ln(-13, target, 13, target, color)}${ln(0, target - 13, 0, target + 13, color)}`;
        else if (base === "z" || base === "p")
          gate +=
            dot(target) +
            (base === "p"
              ? txt(17, (ys[0] + target) / 2 + 7, "P", 23, color, "start")
              : "");
        else gate += box(target, base);
      } else if (o.name === "swap") {
        gate +=
          ln(0, Math.min(...ys), 0, Math.max(...ys), color, 2.5) +
          ys
            .map(
              (yy) =>
                ln(-11, yy - 11, 11, yy + 11, color, 3) +
                ln(-11, yy + 11, 11, yy - 11, color, 3),
            )
            .join("");
      } else if (ys.length > 1) {
        const lo = Math.min(...ys),
          hi = Math.max(...ys);
        gate += `<rect x="-28" y="${lo - 25}" width="56" height="${hi - lo + 50}" rx="6" fill="${fill}" stroke="${color}" stroke-width="${hot ? 4 : 2.3}"/><text x="0" y="${(lo + hi) / 2 + 9}" text-anchor="middle" font-size="27" fill="${color}" font-family="STIX Two Math,serif">${gateLabel(o.name)}</text>`;
      } else if (o.name === "measure") {
        gate += `<rect x="-27" y="${ys[0] - 25}" width="54" height="50" rx="5" fill="${fill}" stroke="${color}" stroke-width="2.3"/><path d="M-18 ${ys[0] + 11}A18 18 0 0 1 18 ${ys[0] + 11}M0 ${ys[0] + 11}L13 ${ys[0] - 13}" fill="none" stroke="${color}" stroke-width="2.8"/>`;
      } else gate += box(ys[0], o.name);
      if (o.name === "measure")
        (o.clbits || []).forEach((b) => measured.set(b, { x: xx, y: ys[0] }));
      o.region
        .filter((r) => r.name === "if_else")
        .forEach((r, depth) => {
          (r.condition_bits || []).forEach((bit) => {
            const m = measured.get(bit),
              bounds = regions.get(r.id);
            if (!m || !bounds) return;
            const token = `${r.id}-${bit}`,
              bend = top + c.qubits.length * dy + depth * 14;
            const first = !feedback.has(token);
            feedback.add(token);
            body += `<path data-morph="${key}-classical-${o.id}-${bit}" data-condition-bit="${bit}" data-feedback-to="${o.id}" d="M${m.x} ${m.y + 25}V${bend}H${bounds.first - 28}V${Math.min(...ys)}" fill="none" stroke="#087d92" stroke-width="2.5" opacity="${first ? 1 : 0}"/>`;
          });
        });
      body += `<g data-morph="${key}-op-${o.id}" data-column="${o.column}" data-circuit-row="${band(o)}" data-operation="${esc(o.name)}" transform="translate(${xx} 0)" class="${hot ? "gate-highlight" : ""}" opacity="${o.index < reveal ? 1 : 0}">${gate}<title>${esc(o.name)} ${esc((o.parameters || []).join(", "))}</title></g>`;
    });
    if (!full && firstColumn + columns <= maximum)
      body += txt(width - 15, top + 8, "…", 30);
    if (label) body += txt(width / 2, height - 5, label, 23, "#5b7088");
    return `<div class="circuit" data-morph="circuit-${key}" data-op-count="${all.length}" data-fold-rows="${bands}" data-first-column="${firstColumn}" data-left="${first}" data-right="${width - 28}"><svg xmlns="http://www.w3.org/2000/svg" data-morph="${key}-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMinYMid meet" role="img" aria-label="Captured program circuit${bands > 1 ? `, ${bands} rows read left to right, top to bottom` : ""}">${body}</svg></div>`;
  }
  function renderQPE(c) {
    const outer = c.operations.find(
        (operation) => operation.name === "for_loop",
      ),
      body = outer?.blocks?.[0] || [],
      inner = body.find((operation) => operation.name === "for_loop"),
      unitary = inner?.blocks?.[0] || [],
      measurement = body.find((operation) => operation.name === "measure"),
      feedback = body.find((operation) => operation.name === "store"),
      preparation = c.operations.filter((operation) => operation.name === "x");
    // This compact view groups only the captured four-CP controlled-U.
    if (
      c.qubits.length !== 5 ||
      preparation.length !== 4 ||
      c.operations.map((operation) => operation.name).join() !==
        "x,x,x,x,for_loop" ||
      preparation.some(
        (operation, i) => operation.qubits.join() !== String(i + 1),
      ) ||
      !outer ||
      !inner ||
      !measurement ||
      !feedback ||
      body.map((operation) => operation.name).join() !==
        "reset,h,for_loop,p,h,measure,store" ||
      [0, 1, 3, 4, 5].some((index) => body[index].qubits.join() !== "0") ||
      unitary.length !== 4 ||
      unitary.some(
        (operation, i) =>
          operation.name !== "cp" ||
          operation.qubits.join() !== `0,${i + 1}` ||
          operation.parameters[0] !== unitary[0].parameters[0],
      )
    )
      throw new Error(
        "The captured QPE source no longer matches its compact diagram",
      );
    const formula = (value) =>
        esc(value)
          .replaceAll("theta", "θ")
          .replaceAll("pi", "π")
          .replaceAll("*", "×")
          .replaceAll(" - ", " − ")
          .replace(/float\[64\]\(([^)]+)\)/g, "$1")
          .replace(" = ", " ← ")
          .replace(/;$/, ""),
      label = (x, y, value, size = 27, color = "#142b45") =>
        `<text x="${x}" y="${y}" text-anchor="middle" font-size="${size}" fill="${color}">${value}</text>`,
      wireY = (wire) => 86 + wire * 36,
      gate = (operation, x, value) =>
        `<g data-operation="${operation.name}" data-source-line="${operation.source_line}"><rect x="${x - Math.max(26, value.length * 8)}" y="${wireY(operation.qubits[0]) - 17}" width="${Math.max(52, value.length * 16)}" height="34" rx="5" fill="white" stroke="#2f70b8" stroke-width="2"/>${label(x, wireY(operation.qubits[0]) + 9, value)}</g>`,
      power = inner.range.match(/^1:1 << (\d+) - (\w+)$/),
      repetitions = power
        ? `2<tspan baseline-shift="super" font-size="70%">${power[1]}−${esc(power[2])}</tspan>`
        : esc(inner.range),
      angle = formula(unitary[0].parameters[0]);
    let drawing = `<g data-control-flow="for_loop" data-iterations="${outer.iterations}" data-source-line="${outer.source_line}"><rect x="158" y="37" width="950" height="264" rx="14" fill="#eaf2fa" fill-opacity=".4" stroke="#87acd0" stroke-width="2"/>${label(630, 26, `${esc(outer.variable)} = ${esc(outer.range.replace(":", ", …, "))} · ${outer.iterations} Rounds`, 29)}</g>`;
    drawing += c.qubits
      .map(
        (qubit) =>
          `<line x1="70" y1="${wireY(qubit.id)}" x2="1092" y2="${wireY(qubit.id)}" stroke="#b6c9dc" stroke-width="2"/>${label(32, wireY(qubit.id) + 8, esc(qubit.label), 23)}`,
      )
      .join("");
    drawing += preparation
      .map((operation) => gate(operation, 112, "X"))
      .join("");
    drawing += gate(body[0], 204, "|0⟩") + gate(body[1], 282, "H");
    drawing += `<g data-control-flow="for_loop" data-range="${esc(inner.range)}" data-source-line="${inner.source_line}"><rect x="330" y="57" width="302" height="181" rx="10" fill="white" fill-opacity=".7" stroke="#2f70b8" stroke-width="2" stroke-dasharray="7 4"/>${label(480, 268, `Repeat ${repetitions} Times`, 27)}</g>`;
    drawing += unitary
      .map((operation, i) => {
        const x = 364 + i * 76,
          target = wireY(operation.qubits[1]);
        return `<g data-operation="cp" data-source-line="${operation.source_line}"><line x1="${x}" y1="86" x2="${x}" y2="${target}" stroke="#2f70b8" stroke-width="2.5"/><circle cx="${x}" cy="86" r="6" fill="#2f70b8"/><circle cx="${x}" cy="${target}" r="6" fill="#2f70b8"/>${label(x + 20, target - 9, "P", 24, "#2f70b8")}</g>`;
      })
      .join("");
    drawing += label(
      480,
      326,
      `Controlled-U · U = P(${angle})<tspan baseline-shift="super" font-size="75%">⊗4</tspan>`,
      23,
    );
    drawing +=
      gate(body[3], 710, `P(${formula(body[3].parameters[0])})`) +
      gate(body[4], 794, "H");
    drawing += `<g data-operation="measure" data-classical-target="${esc(measurement.classical_target)}" data-source-line="${measurement.source_line}"><rect x="859" y="64" width="52" height="44" rx="6" fill="white" stroke="#2f70b8" stroke-width="2"/><path d="M869 93a16 16 0 0 1 32 0M885 93l11 -18" fill="none" stroke="#2f70b8" stroke-width="2"/>${label(1004, 76, esc(measurement.classical_target), 26, "#087d92")}<path d="M885 109V252" fill="none" stroke="#087d92" stroke-width="2" stroke-dasharray="5 4"/></g>`;
    const initialization = c.classical_initializers.find((statement) =>
      statement.startsWith("float[64] theta ="),
    );
    if (!initialization)
      throw new Error(
        "The QPE diagram requires its captured feedback initializer",
      );
    drawing += `<g data-classical-expression="${esc(feedback.expression)}" data-source-line="${feedback.source_line}"><rect x="676" y="252" width="416" height="40" rx="8" fill="#e8f5f0" stroke="#55a194"/>${label(884, 279, formula(feedback.expression), 24, "#087d92")}</g>${label(885, 326, `${formula(initialization.replace("float[64] ", ""))} Initially · Next-Round Feedback`, 22, "#087d92")}`;
    return `<div class="circuit qpe-nested" data-source-sha256="${esc(c.source_sha256)}"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1140 342" role="img" aria-label="Complete source-derived iterative QPE with an outer loop, nested controlled-U loop, measurement and runtime phase feedback">${drawing}</svg></div>`;
  }
  return { flatten, schedule, render, renderQPE };
})();
