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
      const used = o.qubits.filter((q) => rows.has(q));
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
      span.forEach((q) => free.set(q, column + 1));
      (o.clbits || []).forEach((b) => bits.set(b, column + 1));
      return { ...o, index, column };
    });
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
    } = {},
  ) {
    if (!c)
      return '<p class="note">No circuit representation for this compiler stage.</p>';
    const all = schedule(c);
    const maximum = Math.max(0, ...all.map((o) => o.column));
    const highlighted = new Set(Array.isArray(active) ? active : [active]);
    const anchor =
      camera ??
      all[Array.isArray(active) ? active[0] : active >= 0 ? active : start]
        ?.column ??
      0;
    const firstColumn = Math.max(
      0,
      Math.min(
        anchor - (camera !== null || active >= 0 ? 5 : 0),
        Math.max(0, maximum - limit + 1),
      ),
    );
    const operations = all.filter(
      (o) => o.column >= firstColumn && o.column < firstColumn + limit,
    );
    const columns = Math.min(limit, maximum - firstColumn + 1);
    const structured = all.some((o) => o.region.length);
    const width = 1100,
      pitch = Math.min(102, 900 / columns),
      top = structured ? 75 : 40,
      dy = 60;
    const height = Math.max(
      260,
      top + (c.qubits.length - 1) * dy + (structured ? 120 : 70),
    );
    const row = new Map(c.qubits.map((q, i) => [q.id, i]));
    const y = (q) => top + row.get(q) * dy;
    const x = (o) => 155 + (o.column - firstColumn) * pitch;
    const txt = (xx, yy, t, size = 26, color = "#142b45", anchor = "middle") =>
      `<text x="${xx}" y="${yy}" font-size="${size}" fill="${color}" text-anchor="${anchor}">${esc(t)}</text>`;
    const ln = (x1, y1, x2, y2, color = "#2f70b8", w = 3) =>
      `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"/>`;
    let body = c.qubits
      .map(
        (q) =>
          `<g data-morph="${key}-wire-${q.id}">${ln(118, y(q.id), 1072, y(q.id), "#b6c9dc", 2)}${txt(93, y(q.id) + 8, layout ? `p${layout[row.get(q.id)] ?? q.site ?? q.id}` : q.label, 24, "#526d88", "end")}</g>`,
      )
      .join("");
    const regions = new Map();
    operations.forEach((o) =>
      o.region.forEach((r) => {
        if (!["for_loop", "while_loop"].includes(r.name)) return;
        const span = regions.get(r.id) || { first: x(o), last: x(o), r };
        span.last = x(o);
        regions.set(r.id, span);
      }),
    );
    regions.forEach(({ first, last, r }) => {
      body += `<g data-morph="${key}-region-${r.id}"><rect x="${first - 35}" y="${top - 46}" width="${last - first + 70}" height="${(c.qubits.length - 1) * dy + 70}" rx="12" fill="#eaf2fa" fill-opacity=".35" stroke="#87acd0" stroke-width="2" stroke-dasharray="7 5"/>${txt((first + last) / 2, top - 60, r.name === "while_loop" ? "repeat until success" : `${r.iterations} rounds`, 24, "#2f70b8")}</g>`;
    });
    const measured = new Map();
    operations.forEach((o) => {
      const xx = x(o),
        ys = o.qubits.filter((q) => row.has(q)).map(y),
        hot = highlighted.has(o.index);
      if (!ys.length) return;
      const color = hot ? "#0756a8" : "#2f70b8";
      let gate =
        ys.length > 1
          ? ln(0, Math.min(...ys), 0, Math.max(...ys), color, 3)
          : "";
      if (["cx", "cz"].includes(o.name)) {
        gate += `<circle cx="0" cy="${ys[0]}" r="7" fill="${color}"/>`;
        gate +=
          o.name === "cz"
            ? `<circle cx="0" cy="${ys.at(-1)}" r="7" fill="${color}"/>`
            : `<circle cx="0" cy="${ys.at(-1)}" r="22" fill="white" stroke="${color}" stroke-width="3"/>${ln(-14, ys.at(-1), 14, ys.at(-1), color)}${ln(0, ys.at(-1) - 14, 0, ys.at(-1) + 14, color)}`;
      } else if (o.name === "swap") {
        gate += ys
          .map(
            (yy) =>
              ln(-13, yy - 13, 13, yy + 13, color, 4) +
              ln(-13, yy + 13, 13, yy - 13, color, 4),
          )
          .join("");
      } else {
        gate += ys
          .map((yy) => {
            const box = `<rect x="-25" y="${yy - 24}" width="50" height="48" rx="6" fill="${hot ? "#cee6fc" : "white"}" stroke="${color}" stroke-width="${hot ? 4 : 2.5}"/>`;
            if (o.name === "measure")
              return (
                box +
                `<path d="M-17 ${yy + 10}A17 17 0 0 1 17 ${yy + 10}M0 ${yy + 10}L12 ${yy - 12}" fill="none" stroke="${color}" stroke-width="2.8"/>`
              );
            const name =
              {
                reset: "|0⟩",
                prx: "R",
                r: "R",
                rz: "Rz",
                rx: "Rx",
                ry: "Ry",
                rzz: "ZZ",
                gpi: "GPI",
                gpi2: "GPI₂",
                ms: "MS",
              }[o.name] || o.name.toUpperCase();
            return box + txt(0, yy + 9, name, name.length > 3 ? 23 : 27);
          })
          .join("");
      }
      if (o.name === "measure")
        (o.clbits || []).forEach((b) => measured.set(b, { x: xx, y: ys[0] }));
      const condition = o.region.find((r) => r.name === "if_else");
      if (condition) {
        const m = (condition.condition_bits || [])
          .map((b) => measured.get(b))
          .find(Boolean);
        if (m) {
          const bend = top + c.qubits.length * dy + 4;
          body += `<path data-morph="${key}-classical-${o.id}" d="M${m.x} ${m.y + 25}V${bend}H${xx}V${ys[0] + 26}" fill="none" stroke="#087d92" stroke-width="3"/>`;
          body += `<circle cx="${xx}" cy="${ys[0] + 27}" r="5" fill="#087d92"/>${txt((m.x + xx) / 2, bend + 28, "= 1", 23, "#087d92")}`;
        } else gate += txt(0, ys.at(-1) + 50, "if 1", 22, "#087d92");
      }
      body += `<g data-morph="${key}-op-${o.id}" data-column="${o.column}" data-operation="${esc(o.name)}" transform="translate(${xx} 0)" class="${hot ? "gate-highlight" : ""}">${gate}<title>${esc(o.name)} ${esc((o.parameters || []).join(", "))}</title></g>`;
    });
    if (firstColumn + columns <= maximum) body += txt(1085, top + 8, "…", 32);
    body += txt(
      550,
      height - 5,
      label ||
        (firstColumn || maximum >= limit
          ? `Circuit layers ${Math.ceil(firstColumn) + 1}–${Math.floor(firstColumn + columns)} of ${maximum + 1}`
          : "Parallel gates share a column · loop bodies shown once"),
      22,
      "#5b7088",
    );
    return `<div class="circuit" data-op-count="${all.length}"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Captured program circuit">${body}</svg></div>`;
  }
  return { flatten, schedule, render };
})();
