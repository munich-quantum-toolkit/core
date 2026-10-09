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
        body = `<ellipse cx="35" cy="47" rx="29" ry="34" transform="rotate(-25 35 47)" fill="${blue}" opacity=".12"/><ellipse cx="65" cy="47" rx="29" ry="34" transform="rotate(25 65 47)" fill="${cyan}" opacity=".2"/><path d="M33 52L67 39" stroke="${navy}" stroke-width="5"/><circle cx="33" cy="52" r="13" fill="${blue}"/><circle cx="67" cy="39" r="13" fill="${cyan}"/><circle cx="29" cy="47" r="4" fill="white" opacity=".7"/><circle cx="63" cy="34" r="4" fill="white" opacity=".7"/>`;
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

  function ecosystem(step = 99) {
    const hardware =
      icon("cryostat", 1325, 15, 150) +
      icon("iontrap", 1310, 230, 170) +
      icon("neutralatom", 1320, 435, 160) +
      text(1400, 199, "Superconducting", 30) +
      text(1400, 424, "Trapped ions", 30) +
      text(1400, 634, "Neutral atoms", 30);
    const tangle = [
      "M320 150C650 150 825 120 1290 120",
      "M320 150C590 150 710 333 1290 333",
      "M320 150C640 150 930 540 1290 540",
      "M320 495C750 495 770 120 1290 120",
      "M320 495C670 495 870 333 1290 333",
      "M320 495C610 495 860 540 1290 540",
    ]
      .map((d) => path(d, "#A0B9D1", 3))
      .join("");
    return scene(
      "ecosystem",
      "Researchers and HPC systems connect to different quantum technologies through a shared software stack",
      icon("researcher", 65, 25, 235) +
        text(185, 288, "Researchers", 36) +
        icon("server", 100, 350, 190) +
        text(190, 591, "HPC systems", 36) +
        reveal(step, 1, hardware) +
        `<g opacity="${step === 1 ? 1 : 0}">${tangle}</g>` +
        reveal(
          step,
          2,
          `${path("M325 155C520 155 505 310 640 310", blue, 5)}${path("M325 490C520 490 505 340 640 340", blue, 5)}
        ${path("M990 325H1130M1130 120V545M1130 120H1290M1130 335H1290M1130 545H1290", blue, 5)}
        <path d="M620 254Q805 176 990 254V389Q805 467 620 389Z" fill="${pale}"/>
        ${logo("mqss", 670, 240, 260, 94)}${text(805, 373, "Shared software", 38)}${text(805, 418, "stack", 38)}`,
        ) +
        reveal(
          step,
          3,
          `${text(805, 90, "Connect once. Compose the workflow.", 39)}
        <circle cx="535" cy="248" r="10" fill="${cyan}"/><circle cx="1070" cy="325" r="10" fill="${blue}"/><circle cx="1192" cy="120" r="10" fill="${cyan}"/>
        ${text(805, 565, "Programs → devices → results", 34)}`,
        ),
    );
  }

  function stack(step = 99) {
    const layers = [
      { y: 465, name: "Device access", color: "#D2E3F4", at: 1 },
      { y: 345, name: "Runtime & scheduling", color: "#AFCFEA", at: 2 },
      { y: 225, name: "Compiler infrastructure", color: blue, at: 3 },
    ];
    return scene(
      "stack",
      "Application, compiler, runtime and device access form a shared quantum software stack",
      icon("researcher", 35, 190, 235) +
        text(160, 470, "Application", 36) +
        icon("cryostat", 1370, 20, 145) +
        icon("iontrap", 1350, 220, 170) +
        icon("neutralatom", 1360, 430, 160) +
        path("M285 325H410", blue, 5, 'marker-end="url(#stack-arrow)"') +
        layers
          .map(({ y, name, color, at }) =>
            reveal(
              step,
              at,
              `
        <path d="M465 ${y}L895 ${y - 58}L1125 ${y + 6}L695 ${y + 67}Z" fill="${color}"/>
        <path d="M465 ${y}V${y + 24}L695 ${y + 90}V${y + 67}Z" fill="${color}" opacity=".8"/>
        <path d="M695 ${y + 67}V${y + 90}L1125 ${y + 30}V${y + 6}Z" fill="${color}" opacity=".6"/>
        ${text(795, y + 13, name, 36, "middle", `fill="${at === 3 ? "white" : navy}"`)}
      `,
            ),
          )
          .join("") +
        reveal(
          step,
          1,
          path(
            "M1145 484H1225M1225 104V525M1225 104H1340M1225 310H1340M1225 525H1340",
            blue,
            5,
          ),
        ) +
        reveal(
          step,
          3,
          `${logo("mqt", 635, 53, 300, 100)}${text(805, 632, "Shared interfaces. Distinct responsibilities.", 34)}`,
        ),
    );
  }

  function molecule(step = 99) {
    const contours = [1, 0.85, 0.7, 0.55, 0.4]
      .map(
        (scale, i) =>
          `<g transform="translate(435 288) scale(${scale})"><path d="M-290 0C-290-156-90-194 0-98C90-194 290-156 290 0C290 156 90 194 0 98C-90 194-290 156-290 0Z" fill="${i ? "none" : pale}" stroke="${blue}" stroke-width="${i ? 2 : 0}" opacity="${0.17 + i * 0.09}"/></g>`,
      )
      .join("");
    return scene(
      "molecule",
      "A schematic hydrogen molecule and an energy curve illustrate the molecular ground-state problem",
      reveal(
        step,
        0,
        contours +
          path("M310 325L560 240", navy, 17) +
          `<circle cx="310" cy="325" r="58" fill="${blue}"/><circle cx="560" cy="240" r="58" fill="${cyan}"/>
        <circle cx="290" cy="305" r="14" fill="white" opacity=".65"/><circle cx="540" cy="220" r="14" fill="white" opacity=".65"/>` +
          text(310, 424, "H", 44) +
          text(560, 159, "H", 44) +
          text(425, 554, "Molecule + electrons", 38),
      ) +
        reveal(
          step,
          1,
          `${path("M960 490V110M960 490H1485", navy, 3)}${text(1215, 563, "Bond distance", 32)}
        ${text(900, 305, "Energy", 32, "middle", 'transform="rotate(-90 900 305)"')}
        ${path("M990 120C1010 290 1050 435 1133 426C1240 410 1285 296 1480 278", blue, 6)}
        <circle cx="1133" cy="426" r="9" fill="${blue}"/>`,
        ) +
        reveal(
          step,
          2,
          `${path("M1143 417L1207 260", cyan, 3)}${text(1220, 241, "Ground state", 34, "start")}`,
        ) +
        reveal(
          step,
          3,
          text(800, 632, "Ground-state energy · conceptual illustration", 26),
        ),
    );
  }

  function shadows(step = 99, bits = "measured bits") {
    const wires = [250, 330, 410]
      .map((y) => path(`M65 ${y}H1115`, navy, 3))
      .join("");
    const state = `<path d="M220 180Q300 146 380 180V480Q300 514 220 480Z" fill="${pale}" stroke="${blue}" stroke-width="3"/>
      ${text(300, 305, "Trial", 38)}${text(300, 353, "state", 38)}${text(300, 552, "Prepare", 38)}`;
    const rotation = `<path d="M555 180Q635 146 715 180V480Q635 514 555 480Z" fill="#D5EAF2" stroke="${cyan}" stroke-width="3"/>
      ${text(635, 305, "Random", 36)}${text(635, 353, "basis", 36)}${text(635, 552, "Rotate", 38)}`;
    const meters = [195, 275, 355]
      .map((y) => icon("measurement", 892, y, 105))
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
        ${text(800, 626, "Schematic workflow · numerical results follow", 28)}`,
        ),
    );
  }

  window.MQSF_VIZ = { icon, ecosystem, stack, molecule, shadows, walkers };
})();
