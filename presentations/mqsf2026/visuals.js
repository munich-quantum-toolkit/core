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
  const blue = "#2F70B8";
  const navy = "#142B45";
  const pale = "#E8F1FA";
  const cyan = "#60B3CB";
  const gold = "#C58B42";
  const text = (x, y, value, size = 34, anchor = "middle", extra = "") =>
    `<text x="${x}" y="${y}" font-size="${size}" text-anchor="${anchor}" ${extra}>${value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")}</text>`;
  const reveal = (step, at, body) =>
    `<g opacity="${step >= at ? 1 : 0}" aria-hidden="${step < at}">${body}</g>`;
  const path = (d, color = blue, width = 5, extra = "") =>
    `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" ${extra}/>`;
  const logo = (name, x, y, width, height) =>
    window.MQSF_ASSETS?.[name]
      ? `<image href="${window.MQSF_ASSETS[name]}" x="${x}" y="${y}" width="${width}" height="${height}" preserveAspectRatio="xMidYMid meet"/>`
      : "";

  function icon(name, x = 0, y = 0, size = 100) {
    let body = "";
    switch (name) {
      case "researcher":
        body = `<ellipse cx="50" cy="95" rx="40" ry="4" fill="${pale}"/>
          <path d="M24 87L28 58Q31 48 43 46H59Q73 51 76 67L80 87" fill="${pale}" stroke="${navy}" stroke-width="2"/>
          <path d="M43 46L50 63L59 46M50 63V88M36 57L32 80M68 59L74 80" fill="none" stroke="${blue}" stroke-width="2"/>
          <path d="M40 48L42 40H58L60 48L50 60Z" fill="#F4D9C4"/>
          <ellipse cx="50" cy="28" rx="15" ry="19" fill="#F4D9C4" stroke="${navy}" stroke-width="2"/>
          <path d="M35 28Q30 7 48 7Q70 5 66 31L61 20Q48 23 40 16L38 31" fill="${navy}"/>
          <path d="M36 29H46M54 29H64M46 29H54" stroke="${navy}" stroke-width="1.5"/>
          <rect x="38" y="26" width="10" height="8" rx="3" fill="none" stroke="${navy}"/>
          <rect x="52" y="26" width="10" height="8" rx="3" fill="none" stroke="${navy}"/>
          <path d="M46 39Q50 42 55 38" fill="none" stroke="${navy}"/>
          <path d="M26 71L34 83L45 83M73 72L68 82L59 83" fill="none" stroke="#E7BFA1" stroke-width="7" stroke-linecap="round"/>
          <path d="M30 70H69L65 89H35Z" fill="white" stroke="${blue}" stroke-width="2"/>
          <circle cx="50" cy="79" r="3" fill="${cyan}"/><path d="M29 91H71" stroke="${navy}" stroke-width="3" stroke-linecap="round"/>`;
        break;
      case "server":
        body = `<path d="M8 14L24 5H91L77 14Z" fill="#BBD2E9" stroke="${navy}" stroke-width="2"/>
          <path d="M77 14L91 5V87L77 96Z" fill="#BBD2E9" stroke="${navy}" stroke-width="2"/>
          <rect x="8" y="14" width="69" height="82" rx="2" fill="${pale}" stroke="${navy}" stroke-width="2"/>
          ${[23, 40, 57, 74].map((sy) => `<rect x="14" y="${sy}" width="57" height="13" rx="2" fill="white" stroke="${blue}" stroke-width="1.5"/><path d="M21 ${sy + 4}H46M21 ${sy + 8}H46" stroke="${navy}" stroke-width="1.5"/><circle cx="61" cy="${sy + 6}" r="2" fill="${cyan}"/>`).join("")}
          <path d="M18 91H66" stroke="${navy}" stroke-width="2"/>`;
        break;
      case "cryostat":
        body = `<path d="M19 10H81M26 10V79M74 10V79" fill="none" stroke="#91A7BB" stroke-width="2"/>
          <ellipse cx="50" cy="12" rx="35" ry="8" fill="#DDE7EE" stroke="${navy}" stroke-width="2"/>
          <path d="M15 12V18Q50 32 85 18V12" fill="#B8C9D6" stroke="${navy}" stroke-width="2"/>
          ${[
            [34, 28],
            [53, 22],
            [71, 16],
          ]
            .map(
              ([sy, rx]) =>
                `<path d="M${50 - rx} ${sy - 5}V${sy + 4}Q50 ${sy + 13} ${50 + rx} ${sy + 4}V${sy - 5}" fill="#C99857" stroke="${gold}" stroke-width="1.5"/><ellipse cx="50" cy="${sy - 5}" rx="${rx}" ry="6" fill="#F2D8AE" stroke="${gold}" stroke-width="1.5"/>`,
            )
            .join("")}
          ${[-12, -4, 4, 12].map((dx) => `<path d="M${50 + dx} 20C${42 + dx} 34 ${58 + dx} 43 ${50 + dx} 57S${42 + dx} 76 ${50 + dx} 87" fill="none" stroke="${gold}" stroke-width="1.7"/>`).join("")}
          <rect x="43" y="85" width="14" height="10" rx="2" fill="${navy}"/><path d="M47 90H53" stroke="${cyan}" stroke-width="2"/>`;
        break;
      case "iontrap":
        body = `<path d="M9 48L36 15L94 48L66 82Z" fill="#D3E2EF" stroke="${navy}" stroke-width="2"/>
          <path d="M9 48V57L66 90L94 57V48M66 82V90" fill="#AFC7DD" stroke="${navy}" stroke-width="2"/>
          ${[0, 1, 2, 3, 4, 5].map((n) => `<path d="M${24 + n * 8} ${34 + n * 4.6}L${35 + n * 8} ${21 + n * 4.6}M${20 + n * 8} ${46 + n * 4.6}L${10 + n * 8} ${59 + n * 4.6}" stroke="${gold}" stroke-width="4"/>`).join("")}
          <path d="M23 40L79 73" stroke="white" stroke-width="7"/>
          ${[0, 1, 2, 3, 4].map((n) => `<circle cx="${31 + n * 9}" cy="${45 + n * 5.3}" r="5" fill="${blue}" opacity=".2"/><circle cx="${31 + n * 9}" cy="${45 + n * 5.3}" r="2.2" fill="${blue}"/>`).join("")}
          <path d="M2 25L84 75" stroke="${cyan}" stroke-width="2" opacity=".65"/>`;
        break;
      case "neutralatom":
        body = `<path d="M12 75L49 96L90 72L53 51Z" fill="${pale}" stroke="${navy}" stroke-width="1.5"/>
          <ellipse cx="51" cy="22" rx="34" ry="12" fill="${pale}" stroke="${blue}" stroke-width="2"/>
          <ellipse cx="51" cy="22" rx="23" ry="7" fill="white" stroke="${blue}"/>
          <path d="M22 29L31 53M80 29L71 53" stroke="${cyan}" stroke-width="1.5" stroke-dasharray="3 3"/>
          ${[0, 1, 2, 3].flatMap((row) => [0, 1, 2, 3].map((col) => `<circle cx="${26 + col * 11 + row * 5}" cy="${63 + row * 6 - col * 3}" r="5" fill="${blue}" opacity=".14"/><circle cx="${26 + col * 11 + row * 5}" cy="${63 + row * 6 - col * 3}" r="2.5" fill="${blue}"/>`)).join("")}`;
        break;
      case "chip":
        body = `<rect x="24" y="24" width="52" height="52" rx="7" fill="${pale}" stroke="${navy}" stroke-width="3"/>
          ${[32, 44, 56, 68].map((p) => `<path d="M${p} 13V24M${p} 76V87M13 ${p}H24M76 ${p}H87" stroke="${navy}" stroke-width="3"/>`).join("")}
          <ellipse cx="50" cy="50" rx="22" ry="9" fill="none" stroke="${blue}" stroke-width="2"/>
          <ellipse cx="50" cy="50" rx="22" ry="9" transform="rotate(60 50 50)" fill="none" stroke="${blue}" stroke-width="2"/>
          <ellipse cx="50" cy="50" rx="22" ry="9" transform="rotate(120 50 50)" fill="none" stroke="${blue}" stroke-width="2"/><circle cx="50" cy="50" r="4" fill="${blue}"/>`;
        break;
      case "photonics":
        body = `<path d="M9 38L42 17L93 47L61 70Z" fill="${pale}" stroke="${navy}" stroke-width="2"/>
          <path d="M9 38V50L61 82L93 59V47M61 70V82" fill="#BED6EB" stroke="${navy}" stroke-width="2"/>
          <path d="M19 38C34 29 45 42 55 45S72 43 84 49M17 47C29 39 41 52 52 55S69 54 82 60" fill="none" stroke="${blue}" stroke-width="3"/>
          <path d="M4 25L30 41M4 41L22 51M71 57L97 74" stroke="${cyan}" stroke-width="3"/>
          <circle cx="42" cy="42" r="4" fill="${cyan}"/><circle cx="66" cy="53" r="4" fill="${cyan}"/>`;
        break;
      case "spin":
        body = `<path d="M9 36L41 17L92 47L60 68Z" fill="${pale}" stroke="${navy}" stroke-width="2"/>
          <path d="M9 36V51L60 82L92 62V47M60 68V82" fill="#BED6EB" stroke="${navy}" stroke-width="2"/>
          ${[0, 1, 2].map((n) => `<path d="M${28 + n * 13} ${28 + n * 8}L${48 + n * 13} ${40 + n * 8}L${36 + n * 13} ${48 + n * 8}L${16 + n * 13} ${36 + n * 8}Z" fill="${gold}"/><circle cx="${38 + n * 11}" cy="${42 + n * 6}" r="4" fill="${blue}"/>`).join("")}
          <path d="M40 38V17M36 23L40 17L44 23M65 52V30M61 36L65 30L69 36" fill="none" stroke="${blue}" stroke-width="2"/>`;
        break;
      case "cloud":
        body = `<path d="M22 77C-1 76-2 42 21 40C16 11 55 2 67 28C96 21 111 63 86 76Z" fill="${pale}" stroke="${blue}" stroke-width="2"/>`;
        break;
      case "measurement":
        body = `<path d="M13 74A38 38 0 0 1 87 74" fill="${pale}" stroke="${navy}" stroke-width="3"/>
          <path d="M50 72L74 36M22 61L29 64M34 39L38 46M59 35L57 43M79 57L72 61" stroke="${blue}" stroke-width="3" stroke-linecap="round"/><circle cx="50" cy="72" r="5" fill="${blue}"/>`;
        break;
      case "circuit":
        body = `<path d="M6 29H94M6 51H94M6 73H94M63 29V74" fill="none" stroke="${navy}" stroke-width="2"/><rect x="22" y="17" width="23" height="23" rx="3" fill="${pale}" stroke="${blue}" stroke-width="2"/><path d="M28 22V35M39 22V35M28 28H39" stroke="${blue}" stroke-width="2"/><circle cx="63" cy="29" r="4" fill="${blue}"/><circle cx="63" cy="51" r="9" fill="white" stroke="${blue}" stroke-width="2"/><path d="M57 51H69M63 45V57" stroke="${blue}" stroke-width="2"/>`;
        break;
      case "molecule":
      case "orbitals":
        body = `<ellipse cx="35" cy="47" rx="29" ry="34" transform="rotate(-25 35 47)" fill="${blue}" opacity=".12"/><ellipse cx="65" cy="47" rx="29" ry="34" transform="rotate(25 65 47)" fill="${cyan}" opacity=".2"/><path d="M33 52L67 39" stroke="${navy}" stroke-width="5"/><circle cx="33" cy="52" r="18" fill="${blue}"/><circle cx="67" cy="39" r="11" fill="${cyan}"/><circle cx="29" cy="47" r="4" fill="white" opacity=".7"/><circle cx="63" cy="34" r="4" fill="white" opacity=".7"/>`;
        break;
      default:
        throw new Error(`Unknown MQSF illustration: ${name}`);
    }
    return `<g transform="translate(${x} ${y}) scale(${size / 100})" aria-hidden="true">${body}</g>`;
  }

  function scene(name, label, body) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 650" role="img" aria-label="${label}" class="keynote-illustration" style="font-family:Inter,Arial,sans-serif;fill:${navy}">
      <defs><marker id="${name}-arrow" markerWidth="11" markerHeight="9" refX="9" refY="4.5" orient="auto"><path d="M0 0L10 4.5L0 9" fill="${blue}"/></marker></defs>
      <rect width="1600" height="650" fill="white"/>${body}</svg>`;
  }

  const keyed = (
    key,
    body,
    transform = "translate(0 0) scale(1)",
    opacity = 1,
  ) =>
    `<g data-morph="${key}" transform="${transform}" opacity="${opacity}">${body}</g>`;

  // These scenes share the same tree and keys so a click can move existing objects.
  function landscape(stage, expansion = 0, title = false) {
    const clean = stage >= 3;
    const expanded = expansion >= 1;
    const small = expanded ? 0 : 1;
    const qp = [
      ["cryostat", "Superconducting", 8],
      ["iontrap", "Trapped ions", 125],
      ["neutralatom", "Neutral atoms", 242],
      ["photonics", "Photonics", 359],
      ["spin", "Spin qubits", 476],
    ];
    const users = [
      ["Researchers", "molecule", 8],
      ["Developers", "circuit", 120],
      ["Experimentalists", "measurement", 232],
      ["Computer scientists", "chip", 344],
      ["End users", "orbitals", 456],
    ];
    const userArt = users
      .map(([label, badge, y], i) =>
        keyed(
          `user-${i}`,
          icon("researcher", 33, 0, 82) +
            icon(badge, 110, 35, 40) +
            text(100, 107, label, 26),
          `translate(28 ${y}) scale(1)`,
        ),
      )
      .join("");
    const qpuArt = qp
      .map(([kind, label, y], i) =>
        keyed(
          `qpu-${i}`,
          icon(kind, 0, 0, 104) + text(120, 64, label, 30, "start"),
          `translate(1215 ${y}) scale(1)`,
        ),
      )
      .join("");
    const classical =
      keyed(
        "hpc-racks",
        icon("server", 0, 0, 109) +
          icon("server", 81, 0, 109) +
          icon("server", 162, 0, 109),
        title ? "translate(80 112) scale(1)" : "translate(470 478) scale(1)",
      ) +
      keyed(
        "hpc-label",
        text(title ? 217 : 607, title ? 264 : 630, "HPC centers", 32),
      ) +
      keyed(
        "cloud-icon",
        icon("cloud", 0, 0, 124),
        title ? "translate(155 340) scale(1)" : "translate(846 475) scale(1)",
      ) +
      keyed(
        "cloud-label",
        text(title ? 217 : 908, title ? 495 : 630, "Hyperscalers", 32),
      );
    const routes = users
      .flatMap(([, , y], i) =>
        qp.map(([, , qy], j) => {
          const endY = qy + 50;
          const d = `M225 ${y + 47}C${450 + 80 * j} ${y + 20 + 18 * j} ${980 - 90 * i} ${endY + 65 * (i - 1)} 1202 ${endY}`;
          return path(
            d,
            "#AEC5D9",
            2.7,
            `data-morph="route-${i}-${j}" opacity="${stage === 2 ? 0.72 : 0}"`,
          );
        }),
      )
      .join("");
    const inRoutes = users
      .map(([, , y], i) =>
        path(
          `M225 ${y + 47}C380 ${y + 47} 410 306 538 306`,
          blue,
          4,
          `data-morph="input-route-${i}" opacity="${clean && !expanded && !title ? 1 : 0}"`,
        ),
      )
      .join("");
    const outRoutes = qp
      .map(([, , y], i) =>
        path(
          `M1050 309C1155 309 1132 ${y + 50} 1202 ${y + 50}`,
          blue,
          4,
          `data-morph="output-route-${i}" opacity="${clean && !expanded ? 1 : 0}"`,
        ),
      )
      .join("");
    const stackTransform = expanded
      ? "translate(282 52) scale(1.4)"
      : "translate(515 108) scale(0.82)";
    const layouts = expanded
      ? [
          [0, 76, 185, 270],
          [205, 76, 330, 112],
          [205, 210, 330, 136],
          [555, 76, 185, 270],
        ]
      : [
          [20, 93, 660, 65],
          [20, 167, 660, 65],
          [20, 241, 660, 65],
          [20, 315, 660, 65],
        ];
    const layerNames = ["Frontends", "Resources", "Compiler", "Backends"];
    const layerKeys = ["frontends", "resources", "compiler", "backends"];
    const layers = layouts
      .map(([x, y, w, h], i) => {
        const emphasize = expansion !== 2 || i === 2;
        const labelY = expanded ? (i === 1 ? 38 : 44) : 44;
        const color = i === 2 ? "#D5E7F7" : pale;
        const label = text(
          w / 2,
          labelY,
          layerNames[i],
          expanded ? 30 : 38,
          "middle",
          `data-morph="stack-${layerKeys[i]}-label" font-weight="600"`,
        );
        const detail =
          i === 0
            ? logo("pennylane", 41, 86, 104, 48) +
              logo("cuda-q", 45, 155, 96, 41) +
              logo("qir", 53, 213, 82, 33)
            : i === 1
              ? text(w / 2, 82, "Scheduling", 30)
              : i === 2
                ? logo("mqt", 88, 67, 158, 58)
                : logo("qdmi", 34, 105, 119, 64) + icon("chip", 69, 194, 49);
        return keyed(
          `stack-${layerKeys[i]}`,
          `<rect data-morph="stack-${layerKeys[i]}-surface" x="0" y="0" width="${w}" height="${h}" rx="5" fill="${color}" stroke="${blue}" stroke-width="1.7"/>` +
            label +
            keyed(
              `stack-${layerKeys[i]}-detail`,
              detail,
              "translate(0 0) scale(1)",
              expanded ? 1 : 0,
            ),
          `translate(${x} ${y}) scale(1)`,
          emphasize ? 1 : 0.3,
        );
      })
      .join("");
    const stackArt =
      keyed(
        "stack-brand",
        logo("mqsc", 240, -3, 220, 80),
        expanded ? "translate(20 0) scale(1)" : "translate(0 0) scale(1)",
      ) + layers;
    const frame =
      keyed(
        "world-users",
        userArt,
        expanded ? "translate(-155 0) scale(0.78)" : "translate(0 0) scale(1)",
        title ? 0 : small,
      ) +
      keyed(
        "world-qpus",
        qpuArt,
        expanded ? "translate(670 0) scale(0.78)" : "translate(0 0) scale(1)",
        stage >= 1 ? small : 0,
      ) +
      keyed(
        "world-classical",
        classical,
        "translate(0 0) scale(1)",
        stage >= 1 && !expanded ? 1 : 0,
      );
    const titleInput =
      path(
        "M365 230C457 230 432 309 530 309",
        blue,
        4,
        `data-morph="title-input" opacity="${title ? 1 : 0}"`,
      ) +
      path(
        "M294 411C450 411 417 309 530 309",
        blue,
        4,
        `data-morph="title-cloud-input" opacity="${title ? 1 : 0}"`,
      );
    const hpcRoute = path(
      "M730 478C730 445 787 448 787 422",
      blue,
      4,
      `data-morph="classical-input" opacity="${clean && !expanded && !title ? 1 : 0}"`,
    );
    const flow = (key, d, visible) =>
      path(
        d,
        cyan,
        7,
        `data-morph="flow-${key}" class="flow-particles" stroke-dasharray="2 50" opacity="${visible ? 0.9 : 0}"`,
      );
    const flowPaths = keyed(
      "world-flow",
      flow("hpc", "M365 230C457 230 432 309 530 309", title) +
        flow("cloud", "M294 411C450 411 417 309 530 309", title) +
        flow("classical", "M730 478C730 445 787 448 787 422", clean && !title) +
        users
          .map(([, , y], i) =>
            flow(
              `user-${i}`,
              `M225 ${y + 47}C380 ${y + 47} 410 306 538 306`,
              clean && !title,
            ),
          )
          .join("") +
        qp
          .map(([, , y], i) =>
            flow(
              `qpu-${i}`,
              `M1050 309C1155 309 1132 ${y + 50} 1202 ${y + 50}`,
              clean,
            ),
          )
          .join(""),
      "translate(0 0) scale(1)",
      clean && !expanded ? 1 : 0,
    );
    const caption = keyed(
      "stack-caption",
      text(
        800,
        625,
        expansion === 2
          ? "One compiler infrastructure. Many programming models and devices."
          : "Shared interfaces. Distinct responsibilities.",
        32,
      ),
      "translate(0 0) scale(1)",
      0,
    );
    return scene(
      "landscape",
      "Users, HPC and cloud resources connect to diverse quantum hardware through shared software; an expanded view separates frontends, resource management, the compiler and backend interfaces",
      routes +
        inRoutes +
        outRoutes +
        titleInput +
        hpcRoute +
        flowPaths +
        frame +
        keyed("shared-stack", stackArt, stackTransform, clean ? 1 : 0) +
        caption,
    );
  }

  function bridge(step = 3) {
    return landscape(Math.min(3, step), 0, true);
  }

  function ecosystem(step = 3) {
    return landscape(Math.min(3, step));
  }

  function stack(step = 3) {
    return landscape(3, Math.min(3, step));
  }

  function molecule(step = 3) {
    const contours = [1, 0.82, 0.65, 0.48]
      .map(
        (s, i) =>
          `<path data-morph="lih-density-${i}" d="M-265 0C-265-155-70-170 20-82C100-142 242-105 242 0C242 105 100 142 20 82C-70 170-265 155-265 0Z" transform="translate(402 283) scale(${s})" fill="${i ? "none" : pale}" stroke="${blue}" stroke-width="${i ? 2 : 0}" opacity="${0.13 + i * 0.12}"/>`,
      )
      .join("");
    const atoms =
      keyed("lih-bond", path("M270 301L551 263", navy, 14)) +
      keyed(
        "lih-lithium",
        `<circle cx="270" cy="301" r="74" fill="${blue}"/><circle cx="250" cy="278" r="19" fill="white" opacity=".5"/>${text(270, 313, "Li", 41, "middle", 'fill="white" font-weight="600"')}`,
      ) +
      keyed(
        "lih-hydrogen",
        `<circle cx="551" cy="263" r="43" fill="${cyan}"/><circle cx="540" cy="249" r="11" fill="white" opacity=".5"/>${text(551, 275, "H", 36, "middle", 'fill="white" font-weight="600"')}`,
      );
    const orbitals = [0, 1, 2]
      .map((i) => {
        const x = 916 + i * 196;
        return keyed(
          `lih-orbital-${i}`,
          `<ellipse cx="${x - 24}" cy="240" rx="46" ry="79" transform="rotate(-22 ${x - 24} 240)" fill="${blue}" opacity=".16"/><ellipse cx="${x + 24}" cy="240" rx="46" ry="79" transform="rotate(22 ${x + 24} 240)" fill="${cyan}" opacity=".22"/>` +
            text(x, 365, `Orbital ${i + 1}`, 30) +
            path(`M${x - 67} 438H${x - 8}M${x + 8} 438H${x + 67}`, blue, 4) +
            text(x - 39, 491, "α", 31) +
            text(x + 39, 491, "β", 31),
          "translate(0 0) scale(1)",
          step >= 1 ? 1 : 0,
        );
      })
      .join("");
    return scene(
      "molecule",
      "Conceptual lithium hydride molecule with frozen lithium 1s and an active space of two electrons in three spatial orbitals, corresponding to six spin orbitals",
      contours +
        atoms +
        text(400, 487, "Lithium hydride · LiH", 39) +
        keyed(
          "lih-frozen",
          text(400, 548, "Freeze the Li 1s core", 31),
          "translate(0 0) scale(1)",
          step >= 1 ? 1 : 0,
        ) +
        keyed(
          "lih-active-title",
          text(1112, 107, "Active space", 38, "middle", 'font-weight="600"'),
          "translate(0 0) scale(1)",
          step >= 1 ? 1 : 0,
        ) +
        orbitals +
        keyed(
          "lih-space",
          text(1112, 559, "2 electrons · 6 spin orbitals", 33),
          "translate(0 0) scale(1)",
          step >= 2 ? 1 : 0,
        ) +
        keyed(
          "lih-caption",
          text(800, 630, "Conceptual molecular and orbital illustration", 26),
          "translate(0 0) scale(1)",
          step >= 0 ? 1 : 0,
        ),
    );
  }

  function shadows(step = 99, bits = "measured bits") {
    const wires = [190, 240, 290, 340, 390, 440]
      .map((y) => path(`M65 ${y}H1115`, navy, 3))
      .join("");
    const state = `<path d="M220 180Q300 146 380 180V480Q300 514 220 480Z" fill="${pale}" stroke="${blue}" stroke-width="3"/>
      ${text(300, 305, "Trial", 38)}${text(300, 353, "state", 38)}${text(300, 552, "Prepare", 38)}`;
    const rotation = `<path d="M555 180Q635 146 715 180V480Q635 514 555 480Z" fill="#D5EAF2" stroke="${cyan}" stroke-width="3"/>
      ${text(635, 305, "Random", 36)}${text(635, 353, "basis", 36)}${text(635, 552, "Rotate", 38)}`;
    const meters = [155, 205, 255, 305, 355, 405]
      .map((y) => icon("measurement", 911, y, 57))
      .join("");
    const samples = [
      { x: 1250, y: 170, angle: -9, basis: "U₁" },
      { x: 1280, y: 218, angle: -1, basis: "U₂" },
      { x: 1310, y: 270, angle: 8, basis: "U₃" },
    ]
      .map(
        (
          { x, y, angle, basis },
          index,
        ) => `<g transform="translate(${x} ${y}) rotate(${angle} 80 90)">
      <path d="M0 0H137L164 27V180H0Z" fill="white" stroke="${blue}" stroke-width="3"/>
      <path d="M137 0V27H164" fill="${pale}" stroke="${blue}" stroke-width="2"/>
      ${text(82, 65, basis, 36)}${index === 2 ? (bits === "measured bits" ? text(82, 116, "measured", 30) + text(82, 151, "bits", 30) : text(82, 126, bits, 34)) : ""}
    </g>`,
      )
      .join("");
    return scene(
      "shadows",
      "Prepare a quantum trial state, rotate into a random basis, and measure to collect classical shadows",
      wires +
        state +
        text(300, 97, "Quantum work", 36) +
        reveal(step, 1, rotation) +
        reveal(step, 2, meters + text(945, 552, "Measure", 38)) +
        reveal(
          step,
          3,
          `${path("M1115 330H1200", blue, 5, 'marker-end="url(#shadows-arrow)"')}${samples}
        ${text(1370, 552, "Classical shadows", 34)}${text(800, 626, "Repeat with a new basis · keep the basis and the measured bits", 30)}`,
        ),
    );
  }

  function walkers(step = 99) {
    const points = [
      [860, 245],
      [922, 274],
      [989, 246],
      [1070, 280],
      [1151, 258],
      [1214, 307],
      [876, 342],
      [950, 366],
      [1015, 324],
      [1105, 367],
      [1180, 384],
      [1270, 352],
      [885, 440],
      [970, 453],
      [1040, 418],
      [1110, 472],
      [1190, 455],
      [1290, 438],
    ];
    const field = `<path d="M735 510C850 335 937 532 1040 411S1260 365 1390 222" fill="none" stroke="${pale}" stroke-width="80" stroke-linecap="round"/>
      ${[0, 1, 2].map((n) => path(`M735 ${520 - n * 48}C850 ${345 - n * 48} 937 ${542 - n * 48} 1040 ${421 - n * 48}S1260 ${375 - n * 48} 1390 ${232 - n * 48}`, "#C8DBEF", 2)).join("")}`;
    const swarm = points
      .map(
        ([x, y], i) =>
          `<g>${path(`M${x - 42} ${y - 41}Q${x - 24} ${y - 6} ${x} ${y}`, "#A9C9E7", 2)}<circle cx="${x}" cy="${y}" r="${7 + (i % 4) * 2}" fill="${i % 3 ? blue : cyan}" opacity=".85"/></g>`,
      )
      .join("");
    return scene(
      "walkers",
      "Quantum shadows inform a classical ensemble of walkers in auxiliary-field quantum Monte Carlo; the diagram is schematic",
      icon("molecule", 25, 225, 245) +
        text(150, 543, "Molecule", 36) +
        reveal(
          step,
          1,
          `${path("M280 337H377", blue, 5, 'marker-end="url(#walkers-arrow)"')}
        ${icon("circuit", 408, 181, 230)}
        <path d="M440 411H600M451 435H588M465 459H576" stroke="${blue}" stroke-width="5" stroke-linecap="round"/>
        ${text(520, 543, "Quantum shadows", 34)}`,
        ) +
        reveal(
          step,
          2,
          `${path("M650 337H745", blue, 5, 'marker-end="url(#walkers-arrow)"')}${field}${swarm}
        ${text(1055, 130, "Classical walker ensemble", 38)}${text(1060, 543, "Parallel propagation", 34)}`,
        ) +
        reveal(
          step,
          3,
          `${icon("server", 1395, 340, 145)}${text(1465, 543, "HPC", 34)}
        ${text(800, 626, "Schematic workflow · measured results shown separately", 28)}`,
        ),
    );
  }

  window.MQSF_VIZ = {
    icon,
    bridge,
    ecosystem,
    stack,
    molecule,
    shadows,
    walkers,
  };
})();
