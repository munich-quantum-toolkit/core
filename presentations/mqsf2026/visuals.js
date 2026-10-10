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
      case "developer":
      case "user": {
        const skin = name === "developer" ? "#B87D5F" : "#F4D9C4";
        body = `<ellipse cx="50" cy="95" rx="40" ry="4" fill="${pale}"/>
          <path d="M22 89L27 61Q31 47 47 47H57Q73 49 77 63L82 89Z" fill="${name === "developer" ? "#60B3CB" : "#2F70B8"}" stroke="${navy}" stroke-width="2"/>
          <path d="M42 42V50Q50 58 58 50V41" fill="${skin}"/>
          <ellipse cx="50" cy="29" rx="15" ry="19" fill="${skin}" stroke="${navy}" stroke-width="1.5"/>
          ${name === "developer" ? `<path d="M35 30Q24 8 48 7Q72 5 67 36L63 56L57 46L62 29L58 18Q47 28 37 24L37 48L30 56Z" fill="${navy}"/>` : '<path d="M35 23Q33 7 49 8Q66 7 65 23L58 16L46 20L37 18Z" fill="#9BAAB8"/>'}
          <path d="M41 30H44M56 30H59M46 39Q51 43 56 38" fill="none" stroke="${navy}" stroke-width="1.5"/>
          <path d="M29 66L34 83H44M73 66L67 83H58" fill="none" stroke="${skin}" stroke-width="7" stroke-linecap="round"/>
          <path d="M30 69H70L66 89H34Z" fill="white" stroke="${navy}" stroke-width="2"/>
          <circle cx="50" cy="79" r="3" fill="${blue}"/><path d="M28 91H72" stroke="${navy}" stroke-width="3" stroke-linecap="round"/>`;
        break;
      }
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

  // Every mode keeps the same keyed objects; replay changes only their geometry and opacity.
  function architecture(mode = "stack", progress = 1) {
    const p = Math.max(0, Math.min(1, progress ?? 1));
    const ramp = (value, start, end) => {
      const t = Math.max(0, Math.min(1, (value - start) / (end - start)));
      return t * t * (3 - 2 * t);
    };
    const problem = mode === "problem";
    const core = mode === "core";
    const products = ["detail", "ecosystem", "title", "core"].includes(mode);
    const assembly = problem ? 0 : mode === "stack" ? ramp(p, 0, 0.66) : 1;
    const modules = problem ? 0 : mode === "stack" ? ramp(p, 0.68, 0.96) : 1;
    const productTime = mode === "detail" ? p : products ? 1 : 0;
    const qdmi = ramp(productTime, 0.17, 0.3);
    const physical = 1 - ramp(productTime, 0.04, 0.15);
    const compiler = ramp(productTime, 0.46, 0.6);
    const orchestrator = ramp(productTime, 0.79, 0.94);
    const closing = mode === "ecosystem";
    const external = core ? 0 : closing ? ramp(p, 0.2, 0.72) : 1;
    const visible = (start, end) => (problem ? ramp(p, start, end) : 1);
    const mix = (a, b) => a + (b - a) * assembly;
    const users = [
      { name: "Researchers", icon: "researcher", badge: "molecule", y: 73 },
      { name: "Experimentalists", icon: "researcher", badge: "chip", y: 218 },
      {
        name: "Software",
        second: "developers",
        icon: "developer",
        badge: "circuit",
        y: 363,
      },
      { name: "End users", icon: "user", badge: "measurement", y: 508 },
    ];
    const computes = [
      { name: "Workstation", y: 108 },
      { name: "Cloud compute", y: 303 },
      { name: "HPC", y: 498 },
    ].map((compute, i) => ({
      ...compute,
      x: mix(650, 410 + i * 285),
      y: mix(compute.y, 574),
    }));
    const qpus = [
      { kind: "cryostat", name: "Superconducting", y: 103 },
      { kind: "iontrap", name: "Trapped ions", y: 179 },
      { kind: "neutralatom", name: "Neutral atoms", y: 255 },
      { kind: "cryostat", name: "Superconducting", y: 398 },
      { kind: "photonics", name: "Photonics", y: 474 },
      { kind: "spin", name: "Spin qubits", y: 550 },
    ];
    const arrow = 'marker-end="url(#architecture-arrow)"';
    const route = (key, d, opacity, extra = "", color = blue, width = 3) =>
      path(
        d,
        color,
        width,
        `data-morph="${key}" opacity="${opacity}" ${extra}`,
      );
    const module = (key, x, y, width, height, body, opacity = modules) =>
      keyed(
        key,
        `<rect width="${width}" height="${height}" rx="12" fill="#EEF5FB" stroke="#A9C6E1" stroke-width="2"/>${body}`,
        `translate(${x} ${y}) scale(1)`,
        opacity,
      );

    const rawRoutes =
      users
        .flatMap((user, u) =>
          computes.map((compute, c) => {
            const at = u === 0 ? [0.27, 0.58, 0.72][c] : 0.85;
            const reveal = visible(at, at + 0.1);
            return route(
              `world-user-compute-${u}-${c}`,
              `M220 ${user.y}C360 ${user.y} 450 ${compute.y} ${compute.x - 72} ${compute.y}`,
              reveal * (1 - assembly) * external * 0.65,
              `data-route="user-classical" data-from="user-${u}" data-to="compute-${c}" pathLength="1" stroke-dasharray="1" stroke-dashoffset="${1 - reveal}"`,
              "#8AAFCB",
              3,
            );
          }),
        )
        .join("") +
      computes
        .flatMap((compute, c) =>
          qpus.map((qpu, q) => {
            const at =
              c === 0 && q < 3
                ? [0.29, 0.4, 0.48][q]
                : c === 1 && q < 3
                  ? 0.58
                  : c === 2
                    ? 0.72
                    : 0.85;
            const reveal = visible(at, at + 0.1);
            return route(
              `world-compute-qpu-${c}-${q}`,
              `M${compute.x + 72} ${compute.y}C855 ${compute.y} 1010 ${qpu.y} 1152 ${qpu.y}`,
              reveal * (1 - assembly) * external * 0.65,
              `data-route="classical-qpu" data-from="compute-${c}" data-to="qpu-${q}" pathLength="1" stroke-dasharray="1" stroke-dashoffset="${1 - reveal}"`,
              "#8AAFCB",
              3,
            );
          }),
        )
        .join("");
    const userInput = users
      .map((user, i) =>
        route(
          `world-user-frontend-${i}`,
          `M220 ${user.y}C280 ${user.y} 280 167 326 167`,
          modules * external * 0.7,
          `${arrow} data-route="user-classical"`,
        ),
      )
      .join("");
    const deviceOutput = qpus
      .map((qpu, i) =>
        route(
          `world-backend-qpu-${i}`,
          `M1118 496C1140 496 1105 ${qpu.y} 1152 ${qpu.y}`,
          modules * physical * external * 0.7,
          `${arrow} data-route="classical-qpu"`,
        ),
      )
      .join("");
    const hosting = keyed(
      "world-hosting",
      route(
        "hosting-bus",
        "M410 546V538H1130V354H1117M695 546V538M980 546V538",
        1,
        arrow,
        "#739BBE",
        3,
      ),
      "translate(0 0) scale(1)",
      modules * external,
    );
    const people = users
      .map((user, i) =>
        keyed(
          `world-user-${i}`,
          icon(user.icon, 25, 0, 94) +
            icon(user.badge, 114, 44, 38) +
            text(83, user.second ? 112 : 126, user.name, 26) +
            text(83, 147, user.second || "", 26),
          `translate(42 ${user.y - 48}) scale(1)`,
          visible(0, 0.09) * external,
        ),
      )
      .join("");
    const workstation = `<rect x="5" y="6" width="92" height="65" rx="5" fill="${pale}" stroke="${navy}" stroke-width="2"/><rect x="13" y="14" width="76" height="47" rx="2" fill="white" stroke="${blue}"/><path d="M41 72V85H62V72M29 87H74" fill="none" stroke="${navy}" stroke-width="3"/><path d="M23 49L38 32L51 43L71 24" fill="none" stroke="${cyan}" stroke-width="3"/>`;
    const classical = computes
      .map((compute, i) =>
        keyed(
          `world-compute-${i}`,
          (i === 0
            ? keyed("world-workstation-symbol", workstation)
            : i === 1
              ? icon("cloud", 0, 0, 102)
              : icon("server", -9, 0, 88) + icon("server", 42, 0, 88)) +
            text(51, 123, compute.name, mix(27, 39)),
          `translate(${compute.x - 51 * mix(1, 0.64)} ${compute.y - 45 * mix(1, 0.64)}) scale(${mix(1, 0.64)})`,
          visible(0.09, 0.18) * external,
        ),
      )
      .join("");
    const deviceGroups = keyed(
      "world-device-groups",
      `<rect x="1141" y="48" width="448" height="276" rx="16" fill="#F7FAFD" stroke="#C4D6E7" stroke-width="2"/><rect x="1141" y="347" width="448" height="281" rx="16" fill="#F7FAFD" stroke="#C4D6E7" stroke-width="2"/>` +
        text(1160, 81, "Cloud-hosted QPUs", 27, "start") +
        text(1160, 382, "On-premises QPUs", 27, "start"),
      "translate(0 0) scale(1)",
      visible(0.18, 0.27) * physical * external,
    );
    const devices = qpus
      .map((qpu, i) =>
        keyed(
          `world-qpu-${i}`,
          icon(qpu.kind, 0, 0, 65) + text(85, 41, qpu.name, 26, "start"),
          `translate(1157 ${qpu.y - 14}) scale(1)`,
          visible(0.18, 0.27) * physical * external,
        ),
      )
      .join("");
    const headings = keyed(
      "world-column-labels",
      text(126, 31, "Users", 29) +
        text(650, 31, "Classical compute", 29) +
        text(1364, 31, "Quantum processors", 29),
      "translate(0 0) scale(1)",
      (problem ? 1 : mode === "stack" ? 1 - ramp(p, 0, 0.18) : 0) * external,
    );

    const frontends = [
      { name: "Qiskit", brand: "qiskit", program: true, runtime: "Runtime" },
      { name: "OpenQASM 3", program: true, runtime: "via host" },
      { name: "jeff", brand: "jeff", program: true, runtime: "via host" },
      {
        name: "PennyLane",
        brand: "pennylane",
        program: false,
        runtime: "Runtime",
      },
      {
        name: "CUDA-Q",
        brand: "cuda-q",
        program: false,
        runtime: "Planned",
        planned: true,
      },
    ];
    const frontendCards = keyed(
      "stack-frontends",
      text(724, 29, "Frontends & Programming Models", 28) +
        frontends
          .map((frontend, i) =>
            keyed(
              `frontend-${i}`,
              `<rect width="151" height="109" rx="8" fill="${frontend.planned ? "white" : pale}" stroke="${frontend.planned ? "#91A2B5" : "#A9C6E1"}" stroke-width="2" ${frontend.planned ? 'stroke-dasharray="7 5"' : ""}/>` +
                (frontend.brand === "pennylane"
                  ? logo(frontend.brand, 13, 7, 125, 33)
                  : frontend.brand
                    ? logo(frontend.brand, 12, 7, 31, 31) +
                      text(96, 32, frontend.name, 24)
                    : text(75, 32, frontend.name, 23)) +
                text(
                  69,
                  66,
                  frontend.planned ? "Planned" : "Program",
                  23,
                  "middle",
                  `fill="${frontend.program ? blue : "#8394A5"}"`,
                ) +
                text(
                  69,
                  97,
                  frontend.runtime,
                  23,
                  "middle",
                  `fill="${frontend.planned ? "#8394A5" : blue}"`,
                ) +
                `<circle cx="142" cy="60" r="4" fill="${frontend.program ? blue : "white"}" stroke="${frontend.program ? blue : "#8394A5"}" stroke-width="2"/><circle cx="142" cy="91" r="4" fill="${frontend.planned ? "white" : blue}" stroke="${frontend.planned ? "#8394A5" : blue}" stroke-width="2"/>`,
              `translate(${328 + i * 160} 47) scale(1)`,
            ),
          )
          .join("") +
        frontends
          .map((frontend, i) =>
            route(
              `frontend-program-${i}`,
              `M${470 + i * 160} 107H${483 + i * 160}V167H505V224`,
              frontend.program || frontend.planned ? 1 : 0,
              `${arrow} data-lane="program" data-support="${frontend.planned ? "planned" : frontend.program ? "supported" : "unsupported"}" ${frontend.planned ? 'stroke-dasharray="6 5"' : ""}`,
              frontend.planned ? "#91A2B5" : blue,
              2.5,
            ),
          )
          .join("") +
        frontends
          .map((frontend, i) =>
            route(
              `frontend-runtime-${i}`,
              `M${470 + i * 160} 138H${486 + i * 160}V185H968V224`,
              1,
              `${arrow} data-lane="runtime" data-support="${frontend.planned ? "planned" : frontend.runtime === "via host" ? "host" : "supported"}" ${frontend.planned || frontend.runtime === "via host" ? 'stroke-dasharray="6 5"' : ""}`,
              frontend.planned ? "#91A2B5" : blue,
              2.5,
            ),
          )
          .join("") +
        `<rect x="363" y="174" width="298" height="30" fill="white"/><rect x="822" y="193" width="176" height="29" fill="white"/>` +
        text(512, 197, "Program representation", 25) +
        text(909, 217, "Runtime API", 25),
      "translate(0 0) scale(1)",
      modules * (core ? 0 : closing ? ramp(p, 0.25, 0.55) : 1),
    );
    const boundary = keyed(
      "stack-core-boundary",
      `<rect x="319" y="210" width="814" height="250" rx="15" fill="#F9FBFE" stroke="${blue}" stroke-width="2.5"/>` +
        keyed(
          "stack-core-brand",
          `<rect x="650" y="193" width="154" height="35" fill="white"/>` +
            text(727, 222, "MQT Core", 26),
          "translate(0 0) scale(1)",
          Math.max(qdmi, compiler, orchestrator),
        ),
      "translate(0 0) scale(1)",
      modules,
    );
    const compilerPanel = module(
      "stack-compiler",
      336,
      227,
      356,
      150,
      keyed(
        "stack-compiler-generic",
        text(178, 31, "Compiler", 27) + text(178, 64, "Infrastructure", 27),
        "translate(0 0) scale(1)",
        1 - ramp(productTime, 0.34, 0.45),
      ) +
        keyed(
          "stack-compiler-product",
          logo("mqt", 14, 20, 77, 49) +
            text(222, 31, "MQT Compiler", 26) +
            text(222, 64, "Collection", 26),
          "translate(0 0) scale(1)",
          compiler,
        ) +
        keyed(
          "stack-compiler-formats",
          text(178, 101, "OpenQASM 3 · QIR 2", 25) +
            text(178, 135, "Qiskit · jeff", 25),
          "translate(0 0) scale(1)",
          1 - ramp(productTime, 0.34, 0.45),
        ) +
        keyed(
          "stack-compiler-integrations",
          logo("mlir", 31, 97, 79, 42) +
            logo("llvm", 144, 97, 80, 42) +
            logo("qir", 256, 97, 63, 42),
          "translate(0 0) scale(1)",
          compiler,
        ),
    );
    const orchestratorPanel = module(
      "stack-orchestrator",
      753,
      227,
      364,
      150,
      keyed(
        "stack-orchestrator-generic",
        text(182, 46, "Resource Management", 25) +
          text(182, 79, "& Orchestration", 28),
        "translate(0 0) scale(1)",
        1 - ramp(productTime, 0.65, 0.78),
      ) +
        keyed(
          "stack-orchestrator-product",
          logo("mqsc", 13, 23, 90, 49) +
            text(240, 44, "MQSC", 29) +
            text(240, 77, "Orchestrator", 26),
          "translate(0 0) scale(1)",
          orchestrator,
        ) +
        keyed(
          "stack-orchestrator-description",
          text(182, 111, "Compile · submit · retrieve", 24),
          "translate(0 0) scale(1)",
          1 - ramp(productTime, 0.65, 0.78),
        ) +
        keyed(
          "stack-orchestrator-integrations",
          logo("slurm", 33, 104, 122, 36) + logo("qdmi", 205, 104, 118, 36),
          "translate(0 0) scale(1)",
          orchestrator,
        ),
    );
    const compilerFlow = keyed(
      "stack-compiler-flow",
      route(
        "compile-request",
        "M932 377V393H514V380",
        1,
        `${arrow} data-flow="compile-request"`,
      ) +
        route(
          "compiled-program",
          "M542 377V429H961V380",
          1,
          `${arrow} data-flow="compiled-program"`,
        ) +
        `<rect x="630" y="380" width="200" height="29" fill="#F9FBFE"/><rect x="595" y="414" width="276" height="32" fill="#F9FBFE"/>` +
        text(730, 401, "Compile request", 23) +
        text(733, 437, "Compiled program", 25),
      "translate(0 0) scale(1)",
      modules,
    );
    const backend = module(
      "stack-backends",
      336,
      465,
      781,
      61,
      text(28, 39, "Backend Interface", 29, "start") +
        keyed(
          "stack-backends-product",
          logo("qdmi", 560, 4, 161, 51),
          "translate(0 0) scale(1)",
          qdmi,
        ),
    );
    const backendFlow = keyed(
      "stack-backend-flow",
      route(
        "device-capabilities",
        "M355 465V380",
        1,
        `${arrow} data-flow="device-capabilities"`,
      ) +
        route(
          "job-submission",
          "M1007 377V462",
          1,
          `${arrow} data-flow="submission"`,
        ) +
        route(
          "execution-results",
          "M1092 465V380",
          1,
          `${arrow} data-flow="results"`,
        ) +
        route(
          "frontend-results",
          "M1092 227V180H981",
          1,
          `${arrow} data-flow="runtime-results"`,
        ) +
        text(443, 405, "Device", 25) +
        text(443, 440, "capabilities", 25) +
        text(
          0,
          0,
          "Submit",
          25,
          "middle",
          'transform="translate(994 416) rotate(-90)"',
        ) +
        text(
          0,
          0,
          "Results",
          25,
          "middle",
          'transform="translate(1077 416) rotate(-90)"',
        ),
      "translate(0 0) scale(1)",
      modules,
    );
    const adapters = [
      ["iqm", "QDMI for IQM", "IQM quantum computers"],
      ["aws", "Amazon Braket", "Cloud quantum devices"],
      ["mqt", "DDSIM", "Quantum simulation"],
      ["ibm", "IBM Quantum", "IBM quantum devices"],
    ];
    const adapterPaths = adapters
      .map((_, i) =>
        route(
          `backend-adapter-${i}`,
          `M1117 496C1160 496 1132 ${101 + i * 151} 1180 ${101 + i * 151}`,
          qdmi * external,
          `${arrow} data-route="classical-qpu"`,
        ),
      )
      .join("");
    const adapterCards = adapters
      .map(([brand, name, description], i) =>
        keyed(
          `world-adapter-${i}`,
          `<rect width="399" height="126" rx="12" fill="#F7FAFD" stroke="#A9C6E1" stroke-width="2"/>` +
            logo(brand, 15, 19, 87, 55) +
            text(249, 43, name, 27) +
            text(249, 77, description, 22) +
            text(249, 108, "QDMI implementation", 23),
          `translate(1180 ${39 + i * 151}) scale(1)`,
          qdmi * external,
        ),
      )
      .join("");
    const adapterCredit = keyed(
      "world-adapter-credit",
      text(1380, 644, "Developed & maintained by MQSC", 22),
      "translate(0 0) scale(1)",
      qdmi * external,
    );
    const coreDetail = keyed(
      "core-detail",
      text(729, 56, "Quantum programs, shared infrastructure", 32) +
        text(729, 102, "OpenQASM 3 · QIR 2 · Qiskit · jeff", 29) +
        logo("llvm", 460, 127, 112, 56) +
        logo("mlir", 613, 127, 105, 56) +
        logo("qir", 760, 127, 87, 56) +
        logo("slurm", 891, 127, 128, 56) +
        path(
          "M126 243L75 299M126 243L177 299M75 299L103 348M177 299L103 348M177 299L216 348",
          blue,
          3,
        ) +
        [
          [126, 243],
          [75, 299],
          [177, 299],
          [103, 348],
          [216, 348],
        ]
          .map(
            ([x, y], i) =>
              `<circle cx="${x}" cy="${y}" r="17" fill="${i < 3 ? pale : blue}" stroke="${blue}" stroke-width="2"/>`,
          )
          .join("") +
        text(143, 415, "Quantum IR", 26) +
        text(143, 452, "Decision diagrams", 26) +
        text(143, 489, "ZX calculus", 26) +
        icon("server", 1338, 268, 119) +
        text(1400, 415, "DDSIM", 26) +
        text(1400, 452, "Dynamic execution", 26) +
        text(1400, 489, "Verification", 26) +
        text(
          729,
          600,
          "C++20 · Python · SDK integrations · reusable libraries",
          27,
        ),
      "translate(0 0) scale(1)",
      core ? 1 : closing ? 1 - ramp(p, 0, 0.2) : 0,
    );
    return scene(
      "architecture",
      "Users run programs on workstation, cloud and HPC systems; shared compiler and orchestration infrastructure connects them to quantum devices through a backend interface",
      rawRoutes +
        userInput +
        deviceOutput +
        adapterPaths +
        hosting +
        headings +
        deviceGroups +
        people +
        classical +
        devices +
        boundary +
        frontendCards +
        compilerPanel +
        orchestratorPanel +
        compilerFlow +
        backend +
        backendFlow +
        adapterCards +
        adapterCredit +
        coreDetail,
    );
  }

  function bridge() {
    return architecture("title");
  }

  function ecosystem(step = 3) {
    return architecture(step < 3 ? "problem" : "ecosystem");
  }

  function stack() {
    return architecture("detail");
  }

  function application(
    progress = 1,
    app = window.MQSF_DATA?.application || {},
  ) {
    const p = Math.max(0, Math.min(1, progress));
    const fade = (start) => Math.max(0.14, Math.min(1, (p - start) * 5));
    const count = (value) =>
      value == null ? "" : Number(value).toLocaleString("en-US");
    const track = "M335 205H411M844 205H910M1255 315V385H1096V428M890 513H806";
    const quantum = keyed(
      "app-quantum",
      `<rect x="411" y="56" width="868" height="268" rx="18" fill="#EFF6FC" stroke="#A9C6E1" stroke-width="2"/>` +
        icon("chip", 444, 71, 57) +
        text(519, 110, "Quantum device / simulator", 30, "start") +
        icon("circuit", 441, 153, 105) +
        path("M554 198H597", blue, 3, 'marker-end="url(#application-arrow)"') +
        `<rect x="614" y="159" width="169" height="85" rx="8" fill="white" stroke="${blue}" stroke-width="2"/>` +
        text(698, 193, "Random", 28) +
        text(698, 227, "basis", 28) +
        icon("measurement", 807, 166, 76) +
        path("M884 200H942", blue, 3, 'marker-end="url(#application-arrow)"') +
        text(
          650,
          290,
          `${count(app.workload?.snapshots)} shadow circuits · ${count(app.workload?.shots_per_snapshot)} shots each`,
          24,
        ) +
        [0, 1, 2]
          .map(
            (i) =>
              `<g transform="translate(${970 + i * 21} ${132 + i * 16}) rotate(${i * 5 - 5} 85 51)"><rect width="173" height="101" rx="6" fill="white" stroke="${blue}" stroke-width="2"/>${i === 2 ? text(86, 42, "Basis + bits", 27) : path("M28 35H80M90 35H145", "#A9C6E1", 3)}${path("M28 67H55M65 67H89M101 67H146", cyan, 5)}</g>`,
          )
          .join("") +
        text(1091, 304, "Classical shadows", 27),
      "translate(0 0) scale(1)",
      fade(0.05),
    );
    const classical = keyed(
      "app-classical",
      `<rect x="346" y="421" width="929" height="192" rx="18" fill="#E7F1FA" stroke="#A9C6E1" stroke-width="2"/>` +
        path(
          "M891 516H820M580 516H545",
          blue,
          3,
          'marker-end="url(#application-arrow)"',
        ) +
        icon("server", 1170, 443, 71) +
        text(1147, 480, "Classical CPUs", 29, "end") +
        text(1096, 548, "Parallel AFQMC", 32) +
        text(
          1096,
          584,
          `${count(app.propagation?.walkers)} walkers · ${count(app.propagation?.steps)} steps`,
          25,
        ) +
        [0, 1, 2, 3]
          .map((row) =>
            [0, 1, 2, 3, 4, 5]
              .map(
                (col) =>
                  `<circle cx="${601 + col * 34}" cy="${460 + row * 34}" r="9" fill="${(row + col) % 3 ? blue : cyan}" opacity=".8"/>`,
              )
              .join(""),
          )
          .join("") +
        text(660, 599, "Walker ensemble", 24) +
        path(
          "M845 460C885 393 983 390 1016 428",
          "#16899D",
          3,
          'marker-end="url(#application-arrow)"',
        ) +
        text(914, 401, "Reuse the shadows", 25) +
        text(460, 483, "Energy", 31) +
        text(460, 520, "estimate", 31) +
        text(460, 565, "Weighted average", 23),
      "translate(0 0) scale(1)",
      fade(0.48),
    );
    return scene(
      "application",
      "Quantum-assisted AFQMC: a molecular problem produces shadow circuits; a quantum device or simulator returns classical shadows, then parallel classical CPUs reuse them to propagate a walker ensemble and estimate energy",
      keyed(
        "app-molecule",
        icon("molecule", 111, 122, 151) +
          text(188, 306, "Molecular problem", 32) +
          text(188, 349, "LiH · small active space", 25) +
          text(188, 391, "Trial state + integrals", 25),
      ) +
        path(track, "#A9C6E1", 4) +
        path(
          track,
          blue,
          5,
          `class="flow-particles" stroke-dasharray="8 52" opacity="${p > 0.15 ? 0.8 : 0}"`,
        ) +
        quantum +
        classical +
        keyed(
          "app-data-transfer",
          text(1408, 358, "Quantum data", 27) +
            text(1408, 396, "for classical CPUs", 25),
          "translate(0 0) scale(1)",
          fade(0.35),
        ),
    );
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
    architecture,
    application,
    bridge,
    ecosystem,
    stack,
    molecule,
    shadows,
    walkers,
  };
})();
