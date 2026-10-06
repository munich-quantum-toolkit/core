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
  const data = window.MQSF_DATA || {};
  const $ = (id) => document.getElementById(id);
  const chapters = [
    ["opening", "Start"],
    ["compiler", "Compiler"],
    ["architecture", "Device"],
    ["execution", "Execution"],
    ["application", "Application"],
    ["benchmarks", "Evidence"],
    ["closing", "Takeaways"],
  ];
  const scenarios = data.scenarios || [];
  const state = {
    chapter: 0,
    scenario: 0,
    variant: 0,
    stage: 0,
    output: "stage",
    progress: 0,
    running: false,
    routeTimer: null,
  };
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const svgNS = "http://www.w3.org/2000/svg";
  let graphPositions = new Map();
  let replayFrame = 0;
  let previousFrame = 0;

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function svg(tag, attributes = {}, text) {
    const node = document.createElementNS(svgNS, tag);
    for (const [name, value] of Object.entries(attributes))
      node.setAttribute(name, value);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function option(value, label, disabled = false) {
    const node = element("option", label);
    node.value = value;
    node.disabled = disabled;
    return node;
  }

  function empty(container, title, reason) {
    container.replaceChildren();
    const box = element("div", undefined, "empty-state");
    box.append(element("strong", title), element("p", reason));
    container.append(box);
  }

  function selectedScenario() {
    return scenarios[state.scenario] || {};
  }
  function selectedVariant() {
    return (selectedScenario().variants || [])[state.variant] || {};
  }
  function availableArtifacts(artifacts) {
    return (artifacts || []).filter(
      (artifact) =>
        typeof artifact.code === "string" && !artifact.unavailable_reason,
    );
  }
  function stages() {
    return availableArtifacts(selectedVariant().stages);
  }
  function exports() {
    return availableArtifacts(selectedVariant().exports);
  }
  function execution() {
    const variant = selectedVariant();
    if (state.output === "stage") return variant.execution || {};
    if (variant.executions?.[state.output])
      return variant.executions[state.output];
    const artifact = exports().find((output) => output.id === state.output);
    if (
      artifact?.sha256 &&
      artifact.sha256 === variant.execution?.payload_sha256
    )
      return variant.execution;
    return {
      unavailable_reason: `No execution capture is available for ${artifact?.label || state.output}. Select a captured payload format to replay its execution.`,
    };
  }
  function announce(message) {
    $("announcement").textContent = message;
  }

  function download(content, filename, type = "text/plain") {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const link = element("a");
    link.href = url;
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function navigate(index, focus = false) {
    pause();
    stopRouting();
    state.chapter = Math.max(0, Math.min(chapters.length - 1, index));
    chapters.forEach(([id], i) => {
      $(id).hidden = i !== state.chapter;
      $("chapters").children[i].setAttribute(
        "aria-current",
        i === state.chapter ? "step" : "false",
      );
    });
    $("page-number").textContent =
      `${String(state.chapter + 1).padStart(2, "0")} / ${String(chapters.length).padStart(2, "0")}`;
    $("previous").disabled = state.chapter === 0;
    $("next").disabled = state.chapter === chapters.length - 1;
    if (location.hash !== `#${chapters[state.chapter][0]}`) {
      location.hash = chapters[state.chapter][0];
    }
    if (focus) $("main").focus({ preventScroll: true });
    if (state.chapter === 1) applyCodeFocus($("code"));
    window.scrollTo(0, 0);
    announce(chapters[state.chapter][1]);
  }

  function renderCode(container, artifact) {
    container.replaceChildren();
    if (!artifact) return;
    /// Only the local packager produces highlighted HTML; all other content uses text nodes.
    if (artifact.html) container.innerHTML = artifact.html;
    else container.append(element("pre", artifact.code));
    container.dataset.focusOffset = String(
      Math.max(
        0,
        (artifact.focus_line || 1) - (artifact.excerpt_start_line || 1) - 2,
      ),
    );
    applyCodeFocus(container);
    container.scrollLeft = 0;
    if (!reducedMotion.matches)
      container.animate(
        [
          { opacity: 0.25, transform: "translateY(5px)" },
          { opacity: 1, transform: "translateY(0)" },
        ],
        { duration: 220 },
      );
  }

  function applyCodeFocus(container) {
    if (container.offsetHeight && container.dataset.focusOffset !== undefined) {
      container.scrollTop =
        Number(container.dataset.focusOffset) *
        parseFloat(getComputedStyle(container).lineHeight);
      delete container.dataset.focusOffset;
    }
  }

  function selectedArtifact() {
    return state.output === "stage"
      ? stages()[state.stage]
      : exports().find((artifact) => artifact.id === state.output);
  }

  function compressedBytes(artifact) {
    return Uint8Array.from(atob(artifact.full_code_gzip), (character) =>
      character.charCodeAt(0),
    );
  }

  async function fullCode(artifact) {
    if (!artifact.full_code_gzip) return artifact.code;
    if (!window.DecompressionStream)
      throw new Error(
        "This browser cannot expand the source. Download the compressed original instead.",
      );
    const stream = new Blob([compressedBytes(artifact)])
      .stream()
      .pipeThrough(new DecompressionStream("gzip"));
    return new Response(stream).text();
  }

  function renderArtifact() {
    const artifact = selectedArtifact();
    $("pipeline")
      .querySelectorAll("button")
      .forEach((button, i) => {
        const selected = i === state.stage && state.output === "stage";
        button.setAttribute("aria-pressed", selected ? "true" : "false");
        if (selected)
          button.scrollIntoView({
            block: "nearest",
            inline: "nearest",
            behavior: reducedMotion.matches ? "instant" : "smooth",
          });
      });
    $("download-code").disabled = !artifact;
    $("full-code").hidden = !artifact?.full_code_gzip;
    $("full-code").disabled = false;
    $("full-code").textContent = "Full source";
    $("code-title").textContent = artifact?.label || "Capture pending";
    $("code-language").textContent = artifact?.language || "";
    $("code-size").textContent = artifact
      ? artifact.full_code_gzip
        ? `Lines ${artifact.excerpt_start_line || 1}–${(artifact.excerpt_start_line || 1) + artifact.excerpt_lines - 1} of ${artifact.line_count.toLocaleString()} · full source available`
        : `${artifact.code.split("\n").length.toLocaleString()} lines · complete output`
      : "";
    $("control-flow-stats").replaceChildren();
    if (artifact?.control_flow && artifact.language === "mlir") {
      for (const [key, label] of [
        ["counted_loops", "scf.for"],
        ["conditional_loops", "scf.while"],
        ["quantum_branches", "qco.if"],
      ]) {
        const stat = element("div");
        stat.append(
          element("strong", String(artifact.control_flow[key])),
          element("span", label),
        );
        $("control-flow-stats").append(stat);
      }
      $("control-flow-stats").append(
        element("p", "Static IR operation counts; not execution counts."),
      );
    }
    if (artifact) renderCode($("code"), artifact);
    else
      empty(
        $("code"),
        "No compiler capture yet",
        selectedVariant().unavailable_reason ||
          selectedScenario().unavailable_reason ||
          "Generate the native demonstration to populate this view.",
      );
    const previous =
      state.output === "stage" && state.stage > 0
        ? stages()[state.stage - 1]
        : null;
    $("previous-stage").hidden = !previous;
    $("previous-stage").open = false;
    renderCode($("previous-code"), previous);
    const unavailable = (selectedVariant().exports || []).filter(
      (entry) => entry.unavailable_reason,
    );
    $("export-availability").hidden = !unavailable.length;
    $("export-note").textContent = unavailable
      .map((entry) => `${entry.label}: ${entry.unavailable_reason}`)
      .join(" ");
  }

  function renderCompiler() {
    const variant = selectedVariant();
    $("pipeline").replaceChildren();
    stages().forEach((stage, i) => {
      const button = element("button", stage.label);
      button.dataset.number = String(i + 1).padStart(2, "0");
      button.addEventListener("click", () => {
        state.stage = i;
        state.output = "stage";
        $("output-format").value = "stage";
        renderArtifact();
        resetExecution();
      });
      $("pipeline").append(button);
    });
    $("output-format").replaceChildren(option("stage", "Compiler stages"));
    exports().forEach((artifact) =>
      $("output-format").append(option(artifact.id, artifact.label)),
    );
    $("output-format").value = state.output;
    $("variant-description").textContent =
      variant.summary ||
      variant.description ||
      "These views come from the selected compilation run. Required measurement feedback remains part of the program.";
    renderArtifact();
  }

  function selectScenario(index) {
    state.scenario = index;
    state.variant = 0;
    $("scenario").value = String(index);
    $("variant").replaceChildren();
    const variants = selectedScenario().variants || [];
    variants.forEach((variant, i) =>
      $("variant").append(
        option(i, variant.label, !!variant.unavailable_reason),
      ),
    );
    const firstAvailable = variants.findIndex(
      (variant) => !variant.unavailable_reason,
    );
    state.variant = firstAvailable < 0 ? 0 : firstAvailable;
    $("variant").value = String(state.variant);
    $("scenario-summary").textContent = selectedScenario().summary || "";
    refreshVariant();
  }

  function refreshVariant() {
    pause();
    stopRouting();
    state.stage = 0;
    state.output = "stage";
    state.progress = 0;
    renderCompiler();
    setupLayouts();
    renderExecution();
  }

  function topologyPositions(sites, edges) {
    if (
      sites.every((site) => Number.isFinite(site.x) && Number.isFinite(site.y))
    ) {
      return normalizePositions(
        sites.map((site) => ({ id: site.id, x: site.x, y: site.y })),
      );
    }
    /// Emerald's numbered chains determine a readable topological grid, not chip coordinates.
    const ordered = [...sites].sort((a, b) => a.id - b.id);
    const edgeSet = new Set(
      edges.flatMap(([a, b]) => [`${a}:${b}`, `${b}:${a}`]),
    );
    const rows = [];
    for (const site of ordered) {
      const row = rows.at(-1);
      if (row && edgeSet.has(`${row.at(-1).id}:${site.id}`)) row.push(site);
      else rows.push([site]);
    }
    const positions = new Map();
    rows.forEach((row, y) => {
      let offset = 0;
      if (y > 0) {
        const prior = rows[y - 1];
        const siteIndex = row.findIndex((site) =>
          prior.some((other) => edgeSet.has(`${site.id}:${other.id}`)),
        );
        if (siteIndex >= 0) {
          const anchor = prior.find((site) =>
            edgeSet.has(`${site.id}:${row[siteIndex].id}`),
          );
          offset = positions.get(anchor.id).x - siteIndex;
        }
      }
      row.forEach((site, x) =>
        positions.set(site.id, { id: site.id, x: x + offset, y }),
      );
    });
    const gridValid = edges.every(([a, b]) => {
      const first = positions.get(a);
      const second = positions.get(b);
      return (
        first &&
        second &&
        Math.abs(first.x - second.x) + Math.abs(first.y - second.y) === 1
      );
    });
    if (gridValid) return normalizePositions([...positions.values()]);
    return normalizePositions(
      ordered.map((site, i) => ({
        id: site.id,
        x: Math.cos((i * Math.PI * 2) / ordered.length),
        y: Math.sin((i * Math.PI * 2) / ordered.length),
      })),
    );
  }

  function normalizePositions(positions) {
    if (!positions.length) return new Map();
    const xs = positions.map((site) => site.x);
    const ys = positions.map((site) => site.y);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const scale = Math.min(
      620 / (Math.max(...xs) - minX || 1),
      420 / (Math.max(...ys) - minY || 1),
    );
    const midX = (minX + Math.max(...xs)) / 2;
    const midY = (minY + Math.max(...ys)) / 2;
    return new Map(
      positions.map((site) => [
        site.id,
        { x: 380 + (site.x - midX) * scale, y: 260 + (site.y - midY) * scale },
      ]),
    );
  }

  function renderTopology() {
    const device = data.device || {};
    const sites = device.sites || [];
    const edges = device.edges || [];
    $("device-name").textContent = device.name || "IQM Emerald";
    $("topology-stats").textContent =
      `${sites.length} sites · ${edges.length} couplings`;
    $("native-gates").textContent =
      (device.native_gates || []).join(" · ") || "Capture pending";
    graphPositions = topologyPositions(sites, edges);
    const graph = $("topology");
    graph.replaceChildren();
    graph.append(
      svg("title", {}, "Captured device connectivity; topological layout"),
    );
    edges.forEach(([a, b]) => {
      const from = graphPositions.get(a);
      const to = graphPositions.get(b);
      if (from && to)
        graph.append(
          svg("line", {
            x1: from.x,
            y1: from.y,
            x2: to.x,
            y2: to.y,
            class: "topology-edge",
            "data-edge": `${a}:${b}`,
          }),
        );
    });
    sites.forEach((site) => {
      const position = graphPositions.get(site.id);
      const group = svg("g", {
        transform: `translate(${position.x} ${position.y})`,
      });
      group.append(
        svg(
          "title",
          {},
          `${site.name || `Site ${site.id}`} · physical site ${site.id}`,
        ),
      );
      group.append(
        svg("circle", { r: 14, class: "physical-site" }),
        svg("text", { class: "physical-label" }, site.name || site.id),
      );
      graph.append(group);
    });
    graph.append(svg("g", { id: "logical-qubits" }));
    if (!sites.length)
      graph.append(
        svg(
          "text",
          { x: 380, y: 260, "text-anchor": "middle", fill: "#9ab0c5" },
          "Device capture pending",
        ),
      );
  }

  function setupLayouts() {
    const layout = selectedVariant().layout || {};
    const directApplication =
      !!selectedScenario().application && !layout.initial?.length;
    $("architecture-badge").textContent = directApplication
      ? "Direct DDSIM application capture"
      : "Emerald demo profile";
    $("compilation-target").textContent = directApplication
      ? "Emerald compilation not captured"
      : "IQM Emerald topology";
    $("execution-backend").textContent =
      execution().backend ||
      selectedScenario().device_label ||
      "DDSIM QDMI device";
    $("native-gates").textContent = directApplication
      ? "See captured OpenQASM program"
      : (data.device?.native_gates || []).join(" · ");
    $("target-assumption").textContent = directApplication
      ? "This application capture runs directly on ideal DDSIM. It has not been placed or synthesized for Emerald."
      : "Presentation assumption: reset and unrestricted classical control flow are enabled in the target profile.";
    $("target-note").textContent = directApplication
      ? "The Emerald map is shown for reference only; no mapping is claimed for this application capture."
      : "The physical connectivity is real. Demo capabilities are not a claim about currently supported Emerald hardware execution. DDSIM executes the prepared payload.";
    const select = $("layout-view");
    select.replaceChildren();
    if (layout.initial?.length)
      select.append(option("initial", "Initial placement"));
    if (layout.final?.length) select.append(option("final", "Final placement"));
    (layout.swaps || []).forEach((_, i) =>
      select.append(option(`swap:${i}`, `Routing operation ${i + 1}`)),
    );
    if (!select.children.length)
      select.append(option("none", "No layout captured"));
    $("route-play").disabled = !(layout.swaps || []).length;
    renderLayout();
  }

  function renderLayout() {
    const layout = selectedVariant().layout || {};
    const selection = $("layout-view").value;
    const isSwap = selection.startsWith("swap:");
    const swap = isSwap
      ? (layout.swaps || [])[Number(selection.split(":")[1])]
      : null;
    const pair = Array.isArray(swap) ? swap : swap?.sites || swap?.qubits;
    const mapping = isSwap ? swap?.layout || [] : layout[selection] || [];
    const tokens = $("logical-qubits");
    const existing = new Map(
      [...tokens.children].map((node) => [Number(node.dataset.logical), node]),
    );
    const present = new Set();
    mapping.forEach((physical, logical) => {
      const position = graphPositions.get(physical);
      if (!position) return;
      present.add(logical);
      let token = existing.get(logical);
      if (!token) {
        token = svg("g", { class: "logical-token", "data-logical": logical });
        token.append(
          svg("circle", { r: 17 }),
          svg("text", {}, `q${logical}`),
          svg("title", {}, `Logical qubit ${logical}`),
        );
        tokens.append(token);
      }
      token.style.transform = `translate(${position.x}px, ${position.y}px)`;
    });
    for (const [logical, token] of existing)
      if (!present.has(logical)) token.remove();
    $("topology")
      .querySelectorAll(".topology-edge")
      .forEach((edge) => {
        const [a, b] = edge.dataset.edge.split(":").map(Number);
        edge.classList.toggle(
          "active",
          !!pair &&
            ((a === pair[0] && b === pair[1]) ||
              (a === pair[1] && b === pair[0])),
        );
      });
    $("layout-description").textContent = isSwap
      ? `Captured routing operation ${Number(selection.split(":")[1]) + 1}${pair ? `: sites ${pair[0]} ↔ ${pair[1]}` : ""}. Program order, not an execution trace.${mapping.length ? "" : " Intermediate logical placement was not captured."}`
      : mapping.length
        ? `${mapping.length} logical qubits · ${selection} placement. Geometry is a topological layout of captured couplings.`
        : "No successful target-layout capture is available for this compilation variant.";
  }

  function stopRouting() {
    clearInterval(state.routeTimer);
    state.routeTimer = null;
    $("route-play").textContent = "Replay routing";
  }

  function routeReplay() {
    if (state.routeTimer) return stopRouting();
    const swaps = selectedVariant().layout?.swaps || [];
    if (!swaps.length) return;
    let index = 0;
    $("layout-view").value = "swap:0";
    renderLayout();
    $("route-play").textContent = "Pause routing";
    state.routeTimer = setInterval(() => {
      index += 1;
      if (index >= swaps.length) {
        stopRouting();
        if (selectedVariant().layout?.final?.length)
          $("layout-view").value = "final";
      } else $("layout-view").value = `swap:${index}`;
      renderLayout();
    }, 900);
  }

  function renderSequence() {
    const events = execution().events || [];
    const actors = [
      ...new Set(
        events.flatMap((event) => [event.actor, event.target]).filter(Boolean),
      ),
    ];
    const graph = $("sequence");
    graph.parentElement.scrollTop = 0;
    graph.replaceChildren();
    if (!events.length) {
      graph.setAttribute("viewBox", "0 0 650 250");
      graph.append(
        svg(
          "text",
          { x: 325, y: 125, "text-anchor": "middle", fill: "#9ab0c5" },
          "No QDMI trace captured",
        ),
      );
      return;
    }
    const height = Math.max(260, 80 + events.length * 48);
    graph.setAttribute("viewBox", `0 0 650 ${height}`);
    const xFor = (actor) =>
      85 + actors.indexOf(actor) * (480 / Math.max(1, actors.length - 1));
    const defs = svg("defs");
    const marker = svg("marker", {
      id: "arrow",
      markerWidth: 6,
      markerHeight: 6,
      refX: 5,
      refY: 3,
      orient: "auto",
    });
    marker.append(svg("path", { d: "M0,0 L6,3 L0,6 Z", fill: "#68dcf2" }));
    defs.append(marker);
    graph.append(defs);
    actors.forEach((actor) => {
      const x = xFor(actor);
      graph.append(
        svg("text", { x, y: 25, class: "sequence-actor" }, actor),
        svg("line", {
          x1: x,
          x2: x,
          y1: 40,
          y2: height - 12,
          class: "sequence-lane",
        }),
      );
    });
    events.forEach((event, i) => {
      const from = xFor(event.actor);
      const to = xFor(event.target || event.actor);
      const y = 78 + i * 48;
      const group = svg("g", { class: "sequence-event", "data-index": i });
      const line =
        from === to
          ? svg("path", {
              d: `M${from},${y} h35 v12 h-35`,
              "marker-end": "url(#arrow)",
            })
          : svg("line", {
              x1: from,
              x2: to,
              y1: y,
              y2: y,
              "marker-end": "url(#arrow)",
            });
      group.append(
        line,
        svg(
          "text",
          { x: (from + to) / 2, y: y - 8, class: "sequence-operation" },
          event.operation || "QDMI call",
        ),
      );
      group.append(
        svg(
          "text",
          { x: 6, y: y + 4, class: "sequence-time" },
          `${Number(event.time_ms || 0).toFixed(1)} ms`,
        ),
      );
      group.append(
        svg(
          "title",
          {},
          [
            event.operation,
            event.status,
            typeof event.detail === "string"
              ? event.detail
              : JSON.stringify(event.detail || {}),
          ]
            .filter(Boolean)
            .join(" · "),
        ),
      );
      graph.append(group);
    });
  }

  function renderExecution() {
    const run = execution();
    const available = Array.isArray(run.shots) && run.shots.length > 0;
    $("replay-toggle").disabled = !available && !(run.events || []).length;
    $("replay-reset").disabled = $("replay-toggle").disabled;
    $("replay-position").disabled = $("replay-toggle").disabled;
    $("execution-empty").hidden = available;
    $("execution-empty").textContent = available
      ? ""
      : run.unavailable_reason ||
        "No ordered shot capture is available for this variant. Aggregate counts are not expanded into an invented shot sequence.";
    $("trace-kind").textContent =
      run.trace_kind === "python-observed-api"
        ? "Python-observed API calls"
        : run.trace_kind || run.trace_source || "Captured calls";
    $("execution-format").textContent =
      run.format || "No recorded payload selected";
    $("execution-device").textContent = run.format
      ? run.backend || selectedScenario().device_label || "DDSIM QDMI device"
      : "";
    $("execution-payload-hash").textContent =
      run.payload_sha256 || "Unavailable";
    $("execution-timing").textContent = Number.isFinite(run.duration_ms)
      ? `Recorded execution: ${run.duration_ms.toLocaleString(undefined, { maximumFractionDigits: 2 })} ms${run.duration_scope ? ` (${run.duration_scope})` : ""}. Playback shows calls, then outcomes in captured order.`
      : "Recorded duration unavailable. Playback shows calls, then outcomes in captured order.";
    renderSequence();
    updateReplay();
  }

  function resetExecution() {
    pause();
    state.progress = 0;
    renderExecution();
  }

  function replayCounts(shots, completed) {
    const counts = {};
    for (const shot of shots.slice(0, completed))
      counts[shot] = (counts[shot] || 0) + 1;
    return counts;
  }

  function renderHistogram(shots, completed) {
    const graph = $("histogram");
    graph.replaceChildren();
    const counts = replayCounts(shots, completed);
    const finalCounts = replayCounts(shots, shots.length);
    const keys = Object.keys(finalCounts).sort();
    if (!keys.length) {
      graph.append(
        svg(
          "text",
          { x: 310, y: 175, "text-anchor": "middle", fill: "#9ab0c5" },
          "Ordered shot capture pending",
        ),
      );
      return;
    }
    const max = Math.max(...Object.values(finalCounts), 1);
    const width = Math.max(620, keys.length * 44 + 80);
    graph.setAttribute("viewBox", `0 0 ${width} 360`);
    const chartWidth = width - 80;
    const step = chartWidth / keys.length;
    graph.append(
      svg("line", {
        x1: 50,
        x2: width - 20,
        y1: 292,
        y2: 292,
        class: "hist-axis",
      }),
    );
    [0, max / 2, max].forEach((tick) => {
      const y = 292 - (240 * tick) / max;
      graph.append(
        svg("text", { x: 25, y: y + 4, class: "hist-label" }, Math.round(tick)),
      );
    });
    keys.forEach((key, i) => {
      const count = counts[key] || 0;
      const barHeight = (count / max) * 240;
      const x = 55 + i * step;
      const center = x + step * 0.4;
      const bar = svg("rect", {
        x,
        y: 292 - barHeight,
        width: step * 0.8,
        height: barHeight,
        rx: 3,
        class: `hist-bar${shots[completed - 1] === key ? " latest" : ""}`,
      });
      bar.append(
        svg("title", {}, `${key}: ${count} of ${completed} completed shots`),
      );
      graph.append(
        bar,
        svg(
          "text",
          { x: center, y: 281 - barHeight, class: "hist-count" },
          count,
        ),
        svg(
          "text",
          {
            x: center,
            y: 315,
            class: "hist-label",
            transform: keys.length > 12 ? `rotate(45 ${center} 315)` : "",
          },
          key,
        ),
      );
    });
  }

  function updateReplay() {
    const run = execution();
    const events = run.events || [];
    const shots = run.shots || [];
    /// Without per-shot timestamps, replay call order and shot order as separate phases.
    const callFraction = events.length ? (shots.length ? 0.4 : 1) : 0;
    const eventCount = events.length
      ? Math.min(
          events.length,
          Math.floor((state.progress / callFraction) * events.length),
        )
      : 0;
    const shotFraction =
      callFraction === 1
        ? 0
        : Math.max(0, (state.progress - callFraction) / (1 - callFraction));
    const shotCount = Math.min(
      shots.length,
      Math.floor(shotFraction * shots.length + 1e-8),
    );
    $("sequence")
      .querySelectorAll(".sequence-event")
      .forEach((group, i) => {
        group.classList.toggle("visible", i < eventCount);
        group.classList.toggle("current", i === eventCount - 1);
      });
    if (state.running) {
      const current = $("sequence").querySelector(".current");
      const scroller = $("sequence").parentElement;
      if (
        current &&
        current.getBoundingClientRect().bottom >
          scroller.getBoundingClientRect().bottom - 20
      ) {
        scroller.scrollTop +=
          current.getBoundingClientRect().bottom -
          scroller.getBoundingClientRect().bottom +
          50;
      }
    }
    const event = events[eventCount - 1];
    $("event-detail").textContent = event
      ? `${event.operation}${event.status ? ` · ${event.status}` : ""}${typeof event.detail === "string" ? ` · ${event.detail}` : ""}`
      : "Play to follow the recorded calls.";
    $("replay-position").value = String(Math.round(state.progress * 1000));
    $("shot-progress").textContent =
      `${shotCount.toLocaleString()} / ${shots.length.toLocaleString()} shots`;
    $("last-shot").textContent = shotCount
      ? `Latest: ${shots[shotCount - 1]}`
      : "";
    renderHistogram(shots, shotCount);
    const result = run.summary || run.interpretation;
    $("execution-result").textContent =
      state.progress >= 1
        ? typeof result === "string"
          ? result
          : run.factors
            ? `Recovered factors: ${run.factors.join(" × ")}`
            : "Captured execution complete."
        : "";
  }

  function pause() {
    state.running = false;
    cancelAnimationFrame(replayFrame);
    $("replay-toggle").textContent =
      state.progress >= 1 ? "Replay capture" : "Play capture";
  }

  function play() {
    if (state.running) return pause();
    if (state.progress >= 1) {
      state.progress = 0;
      $("sequence").parentElement.scrollTop = 0;
    }
    state.running = true;
    previousFrame = performance.now();
    $("replay-toggle").textContent = "Pause";
    const step = (now) => {
      if (!state.running) return;
      const speed = Number($("replay-speed").value);
      state.progress = Math.min(
        1,
        state.progress + ((now - previousFrame) / 16000) * speed,
      );
      previousFrame = now;
      updateReplay();
      if (state.progress >= 1) pause();
      else replayFrame = requestAnimationFrame(step);
    };
    replayFrame = requestAnimationFrame(step);
  }

  function renderApplication() {
    const applicationScenarios = scenarios.filter((scenario) =>
      /afqmc|shadow/i.test(`${scenario.id} ${scenario.label}`),
    );
    const container = $("application-content");
    container.replaceChildren();
    if (!applicationScenarios.length) {
      empty(
        container,
        "Application capture in preparation",
        "The AFQMC workflow will use real shadow-collection programs and measured results. No scientific results are claimed by this draft.",
      );
      return;
    }
    applicationScenarios.forEach((scenario) => {
      const button = element("button", `Explore ${scenario.label}`, "primary");
      button.addEventListener("click", () => {
        selectScenario(scenarios.indexOf(scenario));
        navigate(1, true);
      });
      container.append(element("p", scenario.summary), button);
    });
  }

  function renderBenchmarks() {
    const benchmark = data.benchmarks || {};
    const container = $("benchmark-content");
    if (!benchmark.rows?.length) {
      empty(
        container,
        "Final measurements are still to come",
        "The full comparison will report all compiler outcomes, native two-qubit gate counts, and runtime on matched workloads. This draft makes no performance or coverage claims.",
      );
      return;
    }
    const table = element("table", undefined, "benchmark-table");
    const head = element("thead");
    const headings = element("tr");
    ["Compiler", "Compiled / total", "Native 2Q gates", "Runtime (s)"].forEach(
      (heading) => {
        const cell = element("th", heading);
        cell.scope = "col";
        headings.append(cell);
      },
    );
    head.append(headings);
    const body = element("tbody");
    benchmark.rows.forEach((row) => {
      const tr = element("tr");
      [
        row.compiler,
        `${row.compiled} / ${row.total}`,
        row.two_qubit_gates ?? "—",
        row.runtime_s ?? "—",
      ].forEach((value) => tr.append(element("td", String(value))));
      body.append(tr);
    });
    table.append(head, body);
    container.append(
      table,
      element(
        "p",
        benchmark.description ||
          "See downloaded data for benchmark provenance and cohort definitions.",
        "fine-print",
      ),
    );
  }

  chapters.forEach(([id, label], i) => {
    const button = element("button", label);
    button.setAttribute("aria-controls", id);
    button.addEventListener("click", () => navigate(i, true));
    $("chapters").append(button);
  });
  scenarios.forEach((scenario, i) =>
    $("scenario").append(
      option(i, scenario.label, !!scenario.unavailable_reason),
    ),
  );
  $("scenario").addEventListener("change", (event) =>
    selectScenario(Number(event.target.value)),
  );
  $("variant").addEventListener("change", (event) => {
    state.variant = Number(event.target.value);
    refreshVariant();
  });
  $("output-format").addEventListener("change", (event) => {
    state.output = event.target.value;
    renderArtifact();
    resetExecution();
  });
  $("layout-view").addEventListener("change", () => {
    stopRouting();
    renderLayout();
  });
  $("route-play").addEventListener("click", routeReplay);
  $("replay-toggle").addEventListener("click", play);
  $("replay-reset").addEventListener("click", () => {
    pause();
    state.progress = 0;
    $("sequence").parentElement.scrollTop = 0;
    updateReplay();
  });
  $("replay-position").addEventListener("input", (event) => {
    pause();
    state.progress = Number(event.target.value) / 1000;
    updateReplay();
  });
  $("full-code").addEventListener("click", async () => {
    const artifact = selectedArtifact();
    if (!artifact) return;
    const button = $("full-code");
    if (button.textContent === "Show excerpt") {
      renderArtifact();
      return;
    }
    button.disabled = true;
    button.textContent = "Expanding…";
    try {
      const code = await fullCode(artifact);
      if (artifact !== selectedArtifact()) return;
      renderCode($("code"), { code, focus_line: artifact.focus_line });
      $("code-size").textContent =
        `${artifact.line_count.toLocaleString()} lines · complete unmodified source`;
      button.textContent = "Show excerpt";
    } catch (error) {
      if (artifact !== selectedArtifact()) return;
      $("code-size").textContent = error.message;
      button.textContent = "Full source";
    } finally {
      button.disabled = false;
    }
  });
  $("download-code").addEventListener("click", async () => {
    const artifact = selectedArtifact();
    if (!artifact) return;
    const filename = `${selectedScenario().id}-${selectedVariant().id}-${artifact.id}.txt`;
    try {
      download(await fullCode(artifact), filename);
    } catch {
      download(compressedBytes(artifact), `${filename}.gz`, "application/gzip");
      announce("Downloaded the complete source as gzip.");
    }
  });
  $("download-data").addEventListener("click", () =>
    download(
      JSON.stringify(data, null, 2),
      "mqsf-2026-capture.json",
      "application/json",
    ),
  );
  $("previous").addEventListener("click", () =>
    navigate(state.chapter - 1, true),
  );
  $("next").addEventListener("click", () => navigate(state.chapter + 1, true));
  document
    .querySelectorAll(".next-chapter")
    .forEach((button) =>
      button.addEventListener("click", () => navigate(state.chapter + 1, true)),
    );
  window.addEventListener("hashchange", () => {
    const index = chapters.findIndex(([id]) => location.hash === `#${id}`);
    if (index >= 0 && index !== state.chapter) navigate(index);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") $("previous-stage").open = false;
    if (
      event.altKey ||
      event.ctrlKey ||
      event.metaKey ||
      event.target.closest(
        "input, select, button, a, summary, textarea, .code-content, [contenteditable=true]",
      )
    )
      return;
    if (["ArrowRight", "PageDown", " "].includes(event.key)) {
      event.preventDefault();
      navigate(state.chapter + 1);
    }
    if (["ArrowLeft", "PageUp"].includes(event.key)) {
      event.preventDefault();
      navigate(state.chapter - 1);
    }
    if (event.key === "Home") {
      event.preventDefault();
      navigate(0);
    }
    if (event.key === "End") {
      event.preventDefault();
      navigate(chapters.length - 1);
    }
  });
  let savedProgress = 0;
  window.addEventListener("beforeprint", () => {
    pause();
    savedProgress = state.progress;
    state.progress = 1;
    updateReplay();
  });
  window.addEventListener("afterprint", () => {
    state.progress = savedProgress;
    updateReplay();
  });
  const provenance = data.provenance || {};
  $("revision").textContent = provenance.core_revision
    ? `Core ${provenance.core_revision.slice(0, 10)} · ${provenance.generated_at || "recorded capture"}`
    : "MQT Core · capture pending";
  $("provenance").textContent = JSON.stringify(provenance, null, 2);
  renderTopology();
  renderApplication();
  renderBenchmarks();
  selectScenario(
    Math.max(
      0,
      scenarios.findIndex((scenario) => !scenario.unavailable_reason),
    ),
  );
  const initialChapter = chapters.findIndex(
    ([id]) => location.hash === `#${id}`,
  );
  navigate(Math.max(0, initialChapter));
})();
