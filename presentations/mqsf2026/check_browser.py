# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check the clicker keynote at FullHD with all network requests blocked."""

from __future__ import annotations

import argparse
import os
import re
from importlib import import_module
from pathlib import Path
from typing import Any

TITLE = "System Software for Quantum Computing: From the Metal to the User"
DEFAULT_HTML = Path(__file__).resolve().parents[2] / "build/mqsf2026/index.html"
LAYOUT_CHECK = """() => {
    const slide = document.getElementById('slide'), bounds = slide.getBoundingClientRect(), issues = [];
    const visible = node => !node.closest('[hidden], [aria-hidden="true"], .reveal.off') &&
        getComputedStyle(node).visibility !== 'hidden' && node.getBoundingClientRect().width > 0;
    for (const node of slide.querySelectorAll('h1, h2, h3, p, pre, img, svg, .metric, .code-frame')) {
        if (!visible(node) || node.parentElement.closest('svg')) continue;
        const rect = node.getBoundingClientRect();
        const name = node.tagName.toLowerCase() + (node.getAttribute('class') ? '.' +
            node.getAttribute('class').trim().replaceAll(' ', '.') : '');
        const label = (node.textContent || node.getAttribute('alt') || '').trim().slice(0, 70);
        const overflow = Math.max(bounds.left - rect.left, rect.right - bounds.right,
            bounds.top - rect.top, rect.bottom - bounds.bottom);
        if (overflow > 3) issues.push(`${name}: outside slide content by ${Math.round(overflow)}px: ${label}`);
    }
    for (const pre of slide.querySelectorAll('pre')) {
        if (!visible(pre)) continue;
        const frame = pre.closest('.code-frame').getBoundingClientRect();
        const walker = document.createTreeWalker(pre, NodeFilter.SHOW_TEXT);
        let node, overflow = 0;
        while ((node = walker.nextNode())) {
            if (!node.textContent.trim()) continue;
            const range = document.createRange();
            range.selectNodeContents(node);
            for (const rect of range.getClientRects())
                overflow = Math.max(overflow, frame.left - rect.left, rect.right - frame.right,
                    frame.top - rect.top, rect.bottom - frame.bottom);
        }
        if (overflow > 3) issues.push(`code: ${Math.round(overflow)}px text clipped by its frame`);
    }
    return issues;
}"""


EVIDENCE_CHECK = """() => {
    const id = document.querySelector('.batch-grid') ? 'afqmc-execution' :
        document.querySelector('.histogram') ? 'adaptive-execution' : '';
    const clock = document.querySelector('.execution-grid .clock-line > div');
    const fraction = clock ? parseFloat(clock.style.width) / 100 : 1;
    let capture, elapsed, actual, expected;
    if (id === 'adaptive-execution') {
        capture = MQSF_DATA.scenarios.find(s => s.id === 'qpe').variants[0].executions['qir-adaptive'];
        const end = Math.max(capture.completed_ms, ...capture.events.map(e => e.time_ms + e.duration_ms));
        elapsed = fraction * end;
        const shots = capture.shot_events.filter(s => s.time_ms <= elapsed);
        const counts = Array(18).fill(0);
        shots.forEach(s => {const n = parseInt(s.outcome, 2); counts[n < 78 ? 0 : n > 93 ? 17 : n - 77]++;});
        const histogram = document.querySelector('.histogram');
        actual = {count:Number(histogram.dataset.shots), bins:[...histogram.querySelectorAll('[data-count]')]
            .map(bar => Number(bar.dataset.count))};
        expected = {count:shots.length, bins:counts};
    } else if (id === 'afqmc-execution') {
        capture = MQSF_DATA.application.execution;
        const begin = capture.events.find(e => e.operation.includes('try_submit_job'))?.time_ms || 0;
        elapsed = begin + fraction * (capture.duration_ms - begin);
        expected = {count:capture.events.filter(e => Number.isInteger(e.program_index) &&
            e.time_ms + e.duration_ms <= elapsed).length};
        actual = {count:Number(document.querySelector('.batch-grid').dataset.programs)};
    } else return null;
    const event = [...capture.events].reverse().find(e => e.time_ms <= elapsed);
    const checkLine = !!document.querySelector('.code-line.hot');
    return {actual, expected, fraction,
        actualLine:checkLine ? Number(document.querySelector('.code-line.hot .line-no')?.textContent) : null,
        expectedLine:checkLine ? event?.source_line || 1 : null};
}"""

CIRCUIT_CHECK = """() => {
    const {schedule, render, flatten} = MQSF_CIRCUIT, failures = [];
    const parity = MQSF_DATA.scenarios.find(s => s.id === 'parity').variants[0];
    const source = parity.stages.find(s => s.id === 'source').circuit;
    const h = schedule(source).filter(o => o.name === 'h');
    if (h.length < 2 || h[0].column !== h[1].column) failures.push('Independent parity H gates are not parallel');
    const feedback = document.createElement('div');
    feedback.innerHTML = render(source, {limit:20});
    if (!feedback.querySelector('[data-morph*="-classical-"]')) failures.push('Missing measured-bit feedback wire');
    let dependencies = 0, windows = 0;
    function checkWindow(c, options, label) {
        const holder = document.createElement('div');
        holder.innerHTML = render(c, options);
        const active = Array.isArray(options.active) ? options.active : [options.active];
        if (holder.querySelectorAll('.gate-highlight').length !== active.length)
            failures.push(`${label}: active gates missing`);
        for (const gate of holder.querySelectorAll('[data-operation]')) {
            const x = Number(gate.getAttribute('transform').match(/translate\\(([-.\\d]+)/)[1]);
            if (x < 143 || x > 1075) failures.push(`${label}: gate outside wires`);
        }
        windows++;
    }
    for (const s of MQSF_DATA.scenarios) for (const v of s.variants) for (const stage of v.stages || []) {
        if (!stage.circuit) continue;
        const measured = new Map();
        for (const op of schedule(stage.circuit)) {
            for (const r of op.region.filter(r => r.name === 'if_else')) for (const b of r.condition_bits || []) {
                if (!measured.has(b) || measured.get(b) >= op.column)
                    failures.push(`${s.id}: feedback precedes measurement`);
                dependencies++;
            }
            if (op.name === 'measure') for (const b of op.clbits || []) measured.set(b, op.column);
        }
    }
    for (const target of MQSF_DATA.targets) {
        const edges = new Set(target.metadata.edges.flatMap(([a,b]) => [`${a},${b}`, `${b},${a}`]));
        for (const stage of target.compilation.stages.filter(s => s.circuit)) {
            const c = stage.circuit, ops = flatten(c);
            if (['place-and-route','target-native-synthesis'].includes(stage.id)) {
                const sites = new Map(c.qubits.map(q => [q.id, q.site]));
                for (const o of ops.filter(o => o.qubits.length === 2))
                    if (!edges.has(o.qubits.map(q => sites.get(q)).join(',')))
                        failures.push(`${target.id}: non-edge gate`);
            }
            if (!['optimized','place-and-route','target-native-synthesis'].includes(stage.id)) continue;
            for (let active = 0; active < ops.length; active++) {
                checkWindow(c, {active, limit:18, start:Math.max(0,active-8)}, `${target.id}/${stage.id}/${active}`);
            }
            const layers = new Map();
            for (const op of schedule(c)) layers.set(op.column, [...(layers.get(op.column) || []), op.index]);
            for (const [column, active] of layers) {
                checkWindow(c, {active, camera:column + 0.37, limit:18}, `${target.id}/${stage.id}/layer${column}`);
            }
        }
    }
    return {failures:failures.slice(0,20), dependencies, windows};
}"""


REAL_MOTION_CHECK = """async () => {
    const failures = [], samples = [], gaps = [];
    const waitFrames = async duration => {
        const start = await new Promise(requestAnimationFrame);
        while (await new Promise(requestAnimationFrame) - start < duration) {}
    };
    const root = document.createElement('div'); document.body.append(root);
    MQSF_MOTION.replace(root,
        '<svg><g data-morph="hidden" opacity="0"><circle r="20"/></g></svg>', 950);
    for (let i = 0; i < 20; i++) {
        await new Promise(requestAnimationFrame);
        if (Number(getComputedStyle(root.querySelector('g')).opacity) !== 0)
            failures.push('A future SVG build flashed before its reveal');
    }
    MQSF_MOTION.finish();
    MQSF_MOTION.update(root, '<svg><g data-morph="stable"><circle r="2"/></g></svg>');
    const group = root.querySelector('g');
    MQSF_MOTION.update(root, '<svg><g data-morph="stable"><circle r="4"/><text>Added</text></g></svg>');
    if (root.querySelector('g') !== group || root.querySelector('circle').getAttribute('r') !== '4')
        failures.push('Timeline updates recreated a stable SVG node');
    MQSF_MOTION.update(root, '<p>Changed scene</p>');
    if (root.innerHTML !== '<p>Changed scene</p>') failures.push('Changed timeline structure was not replaced');
    const phase = opacity => `<svg><g data-morph="next-phase" opacity="${opacity}"><circle r="4"/></g></svg>`;
    MQSF_MOTION.update(root, phase(1));
    const entrance = root.querySelector('g').getAnimations()[0];
    if (!entrance) failures.push('A new compiler phase appeared without an entrance');
    await new Promise(requestAnimationFrame);
    MQSF_MOTION.update(root, phase(1));
    if (root.querySelector('g').getAnimations()[0] !== entrance)
        failures.push('An entrance restarted on a stable timeline frame');
    MQSF_MOTION.update(root, phase(0));
    if (Number(getComputedStyle(root.querySelector('g')).opacity) !== 0)
        failures.push('An entrance overrode a later hidden state');
    MQSF_MOTION.finish();
    root.remove();
    const targetIndex = MQSF_DECK.slides.findIndex(s => s.id === 'device-switch');
    const targetSlide = MQSF_DECK.slides[targetIndex];
    const positions = () => new Map([...document.querySelectorAll('[data-morph^="device-node-"]')]
        .map(node => {
            // consolidate() rewrites Firefox's attribute and changes the geometry under test.
            const {e, f} = node.transform.baseVal.getItem(0).matrix;
            return [node.dataset.morph, [e, f]];
        }));
    for (let step = 1; step <= targetSlide.builds; step++) {
        MQSF_DECK.go(targetIndex, step); MQSF_DECK.finishMotion();
        const destination = positions();
        MQSF_DECK.go(targetIndex, step - 1); MQSF_DECK.finishMotion();
        const origin = positions();
        const key = [...origin.keys()].find(k => destination.has(k)
            && Math.hypot(...destination.get(k).map((v, i) => v - origin.get(k)[i])) > 5);
        if (!key) { failures.push('No shared target node to test'); continue; }
        const from = origin.get(key), to = destination.get(key);
        const distance = point => Math.hypot(...point.map((v, i) => v - from[i]));
        MQSF_DECK.next();
        if (distance(positions().get(key)) > .001)
            failures.push(`Target ${step} flashed its final topology before morphing`);
        await waitFrames(targetSlide.playbacks[step].transition / 2);
        const middle = positions().get(key), fraction = distance(middle) / distance(to);
        if (!(fraction > 0 && fraction < 1))
            failures.push(`Target ${step} did not morph through an intermediate position: ${fraction}`);
        if (document.querySelector('#slide').dataset.progress !== '0')
            failures.push(`Target ${step} replay overwrote its unfinished topology morph`);
        MQSF_DECK.next();
        if (MQSF_DECK.getState().animating || document.querySelector('#slide').dataset.progress !== '')
            failures.push(`Target ${step} could not finish during its transition`);
        if (Math.abs(distance(positions().get(key)) - distance(to)) > .001)
            failures.push(`Target ${step} did not settle its topology on finish`);
        MQSF_DECK.go(targetIndex, step - 1); MQSF_DECK.finishMotion(); MQSF_DECK.next();
        await new Promise(requestAnimationFrame);
        MQSF_DECK.go(0);
        await waitFrames(targetSlide.playbacks[step].transition + 50);
        if (MQSF_DECK.getState().slide !== 0 || document.querySelector('#slide').dataset.progress !== '')
            failures.push(`Target ${step} transition survived navigation`);
    }
    for (const [index, slide] of MQSF_DECK.slides.entries()) {
        for (const [build, playback] of Object.entries(slide.playbacks)) {
            const step = Number(build);
            const start = () => {
                if (step) MQSF_DECK.go(index, step - 1);
                else MQSF_DECK.go(index - 1, MQSF_DECK.slides[index - 1].builds);
                MQSF_DECK.finishMotion(); MQSF_DECK.next();
            };
            start();
            const heading = document.querySelector('#slide h2, #slide h1');
            if (document.querySelector('#slide').dataset.progress !== '0')
                failures.push(`${slide.id}/${step}: first paint was not the initial frame`);
            if (playback.transition) await waitFrames(playback.transition);
            let previous = 0, previousTime;
            for (let i = 0; i < 14; i++) {
                const now = await new Promise(requestAnimationFrame);
                if (previousTime !== undefined) gaps.push(now - previousTime);
                previousTime = now;
                const value = document.querySelector('#slide').dataset.progress;
                const progress = value === '' ? 1 : Number(value);
                if (!Number.isFinite(progress) || progress < previous || progress > 1)
                    failures.push(`${slide.id}/${step}: invalid progression ${previous} -> ${progress}`);
                if (document.querySelector('#slide h2, #slide h1') !== heading)
                    failures.push(`${slide.id}/${step}: recreated heading during playback`);
                previous = progress;
            }
            if (!(previous > 0)) failures.push(`${slide.id}/${step}: stalled at first frame`);
            samples.push({id:slide.id, step, progress:previous});
            // A cancelled callback must not overwrite the destination slide.
            MQSF_DECK.go(0); await new Promise(requestAnimationFrame);
            if (MQSF_DECK.getState().slide !== 0 || document.querySelector('#slide').dataset.progress !== '')
                failures.push(`${slide.id}/${step}: callback survived navigation`);
            // Repeated starts reproduce the first-frame clock race without a mocked clock.
            for (let i = 0; i < 4; i++) {
                start();
                if (playback.transition) await waitFrames(playback.transition);
                for (let frame = 0; frame < 4; frame++) await new Promise(requestAnimationFrame);
                const value = document.querySelector('#slide').dataset.progress;
                if (value !== '' && !(Number(value) > 0)) failures.push(`${slide.id}/${step}: restart stalled`);
            }
            MQSF_DECK.go(0);
        }
    }
    gaps.sort((a,b) => a-b);
    MQSF_DECK.finishMotion();
    return {failures, samples, frames:gaps.length, p95:gaps[Math.floor(gaps.length * .95)] || 0};
}"""


def check(
    html: Path,
    executable: str | None,
    screenshot: Path | None,
    screenshots_dir: Path | None = None,
    engine: str = "chromium",
) -> None:
    """Check all builds, recorded evidence, print pages, and interrupted motion offline."""
    with import_module("playwright.sync_api").sync_playwright() as playwright:
        browser = getattr(playwright, engine).launch(
            executable_path=executable, args=["--no-sandbox"] if engine == "chromium" else []
        )
        network: list[str] = []
        errors: list[str] = []
        geometry: dict[tuple[int, str], int] = {}

        def route_request(route: Any) -> None:  # ruff: ignore[any-type] - Optional Playwright import.
            if route.request.url.startswith(("http:", "https:")):
                network.append(route.request.url)
                route.abort()
            else:
                route.continue_()

        context = browser.new_context(
            offline=True, viewport={"width": 1920, "height": 1080}, reduced_motion="no-preference"
        )
        context.route("**/*", route_request)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        # Freeze the browser clock so even the short 1x replay can be inspected.
        page.clock.install(time="2026-10-14T07:00:00Z")
        page.goto(html.resolve().as_uri(), wait_until="load")
        page.evaluate("document.fonts.ready")
        page.clock.pause_at("2026-10-14T08:00:00Z")
        assert page.title() == TITLE
        slides = page.evaluate("window.MQSF_DECK.slides")
        assert slides
        assert len({slide["id"] for slide in slides}) == len(slides), "Slide IDs must be unique"
        states = [(index, step) for index, slide in enumerate(slides) for step in range(slide["builds"] + 1)]
        ids = {slide["id"]: index for index, slide in enumerate(slides)}
        expected_mapping = page.evaluate("MQSF_DATA.targets[0].compilation.layout.final.slice(0,6)")
        expected_swaps = page.evaluate("MQSF_DATA.targets[0].compilation.layout.swaps.length")
        probe = page.evaluate(CIRCUIT_CHECK)
        assert not probe["failures"], probe["failures"]
        assert probe["dependencies"]
        assert probe["windows"]

        def assert_state(expected: tuple[int, int]) -> dict[str, Any]:
            current = page.evaluate("window.MQSF_DECK.getState()")
            assert (current["slide"], current["step"]) == expected, (expected, current)
            assert page.locator("#slide").get_attribute("data-slide") == str(expected[0] + 1)
            assert page.locator("#slide").get_attribute("data-step") == str(expected[1])
            return current

        def inspect_evidence() -> dict[str, Any] | None:
            evidence = page.evaluate(EVIDENCE_CHECK)
            if evidence:
                assert evidence["actual"] == evidence["expected"], evidence
                assert evidence["actualLine"] == evidence["expectedLine"], evidence
            return evidence

        def inspect_slide(index: int, step: int) -> None:
            page.evaluate("MQSF_DECK.finishMotion()")
            for issue in page.evaluate(LAYOUT_CHECK):
                geometry.setdefault((index, issue), step)
            assert page.locator("#slide-number").inner_text().startswith(f"{index + 1:02d} / {len(slides)}")
            assert page.locator("#section-label").inner_text().strip()
            assert page.locator("#mqsc-logo").is_visible()
            inspect_evidence()
            if slides[index]["id"] == "structured" and step in {1, 2}:
                source = page.evaluate(
                    """id => {
                    const v = MQSF_DATA.scenarios.find(s => s.id === 'parity').variants
                        .find(v => v.id === 'structured');
                    return [...v.stages, ...v.exports].find(a => a.id === id).code;
                }""",
                    "openqasm3" if step == 1 else "qir-adaptive",
                )
                visible = {line.strip() for line in page.locator(".structure-code .code-text").all_text_contents()}
                pattern = (
                    r"[{}]|\b(?:for|if|while|reset|measure)\b"
                    if step == 1
                    else r"\bcall\b.*(?:__reset__|__mz__|__read_result)|\b(?:br|phi|icmp)\b"
                )
                required = [line.strip() for line in source.splitlines() if re.search(pattern, line)]
                assert required
                missing = [line for line in required if line not in visible]
                assert not missing, f"Structured build {step} hides source control or quantum effects: {missing}"
            if slides[index]["id"] == "routing" and step == 2 and not assert_state((index, step))["animating"]:
                expected = [f"q{i} → {site}" for i, site in enumerate(expected_mapping)]
                assert page.locator(".mapping-legend span").all_text_contents() == expected
            if slides[index]["id"] == "routing" and step in {0, 2} and not assert_state((index, step))["animating"]:
                swaps = int(page.locator(".telemetry strong").inner_text().split()[0])
                assert swaps == (0 if step == 0 else expected_swaps), "SWAP telemetry must follow the routing stage"
            page.evaluate("Promise.all([...document.querySelectorAll('#deck img')].map(i => i.decode()))")
            assets = page.evaluate("""() => [...document.querySelectorAll('#deck img, #deck svg image')].map(i =>
                i.currentSrc || i.getAttribute('href') || i.getAttribute('xlink:href'))""")
            assert assets
            assert all(source.startswith("data:image/svg+xml") for source in assets), (
                f"Slide {index + 1} contains a non-vector or external image"
            )
            assert page.locator("#deck canvas, #deck video").count() == 0
            if screenshots_dir and step == slides[index]["builds"]:
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshots_dir / f"slide-{index + 1:02d}.png"))

        def inspect_progress(index: int, step: int, playback: dict[str, Any]) -> None:
            fractions = page.evaluate(
                """() => {
                const c = document.querySelector('.histogram')
                    ? MQSF_DATA.scenarios.find(s => s.id === 'qpe').variants[0].executions['qir-adaptive']
                    : MQSF_DATA.application.execution;
                const times = c.shot_events?.map(e => e.time_ms) || c.events
                    .filter(e => Number.isInteger(e.program_index)).map(e => e.time_ms + e.duration_ms);
                const duration = c.duration_ms || Math.max(c.completed_ms,
                    ...c.events.map(e => e.time_ms + e.duration_ms));
                const begin = c.shot_events ? 0 :
                    c.events.find(e => e.operation.includes('try_submit_job'))?.time_ms || 0;
                return [0.25,0.5,0.75].map(f => (times[Math.floor(f * times.length)] - begin) / (duration - begin));
            }""",
            )
            last = 0.0
            counts = []
            for fraction in fractions:
                page.clock.fast_forward(round(playback["duration"] * (fraction - last)))
                evidence = inspect_evidence()
                assert evidence is not None
                counts.append(evidence["actual"]["count"])
                for issue in page.evaluate(LAYOUT_CHECK):
                    geometry.setdefault((index, issue), step)
                last = fraction
            assert 0 < counts[0] < counts[-1], f"No growing in-progress results on {slides[index]['id']}: {counts}"

        page.keyboard.press("Home")
        completed = loops = 0
        for position, expected in enumerate(states):
            current = assert_state(expected)
            slide = slides[expected[0]]
            playback = slide["playbacks"].get(str(expected[1]))
            if current["animating"]:
                assert playback
                page.clock.run_for(playback.get("transition", 0) + 32)
                evidence = inspect_evidence()
                if evidence and evidence["fraction"] < 1:
                    inspect_progress(*expected, playback)
                else:
                    page.clock.fast_forward(round(playback["duration"] / 2))
                if playback.get("loop"):
                    loops += 1
                    assert assert_state(expected)["animating"], "A looping scene must remain active"
                else:
                    page.keyboard.press("PageDown")
                    assert not assert_state(expected)["animating"], "First click must finish a finite replay"
                    completed += 1
            inspect_slide(*expected)
            if position + 1 < len(states):
                # For looping scenes this click must finish AND advance in one action.
                page.keyboard.press(("PageDown", "Space", "ArrowRight")[position % 3])
        expected_loops = sum(bool(p.get("loop")) for s in slides for p in s["playbacks"].values())
        expected_completed = sum(not p.get("loop") for s in slides for p in s["playbacks"].values())
        assert (loops, completed) == (expected_loops, expected_completed), (loops, completed)
        page.keyboard.press("PageDown")
        assert_state(states[-1])
        for expected in reversed(states[:-1]):
            page.keyboard.press("PageUp")
            assert not assert_state(expected)["animating"]
        page.keyboard.press("PageUp")
        assert_state(states[0])

        page.keyboard.press("End")
        assert_state(states[-1])
        page.keyboard.press("Home")
        assert_state(states[0])
        for digit in str(len(slides)):
            page.keyboard.press(digit)
        assert page.locator("#jump-indicator").is_visible()
        page.keyboard.press("Enter")
        assert_state((len(slides) - 1, 0))
        assert page.locator("#jump-indicator").is_hidden()
        page.reload(wait_until="load")
        assert_state((len(slides) - 1, 0))
        page.keyboard.press("Home")
        page.keyboard.press("g")
        assert page.locator("#overview").is_visible()
        assert page.locator("#slide-grid button").count() == len(slides)
        page.keyboard.press("ArrowRight")
        page.keyboard.press("ArrowDown")
        assert page.locator("#slide-grid button.selected").get_attribute("data-slide") == "6"
        page.keyboard.press("Enter")
        assert_state((6, 0))
        assert page.locator("#overview").is_hidden()
        page.keyboard.press("Escape")
        page.keyboard.press("ArrowLeft")
        page.keyboard.press("Escape")
        assert page.locator("#overview").is_hidden()
        assert_state((6, 0))
        for key, selector in (("b", "#blackout"), ("p", "#speaker-notes")):
            page.keyboard.press(key)
            assert page.locator(selector).is_visible()
            page.keyboard.press(key)
            assert page.locator(selector).is_hidden()

        page.evaluate("dispatchEvent(new Event('beforeprint'))")
        page.emulate_media(media="print")
        assert page.locator("#print-deck .print-slide").count() == len(slides)
        assert page.locator("#deck").is_hidden()
        for index in range(len(slides)):
            printed = page.locator(".print-slide").nth(index)
            assert printed.locator(".slide-footer").inner_text().endswith(f"{index + 1} / {len(slides)}")
            assert printed.locator("h1, h2").count() > 0
        if engine == "chromium":
            pdf = page.pdf(prefer_css_page_size=True, print_background=True)
            assert len(re.findall(rb"/Type\s*/Page\b", pdf)) == len(slides), "Printed PDF has extra or missing pages"
        page.evaluate("dispatchEvent(new Event('afterprint'))")
        page.emulate_media(media="screen")
        assert page.locator("#print-deck .print-slide").count() == 0
        page.keyboard.press("Home")
        if screenshot:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot))
        context.close()

        context = browser.new_context(offline=True, viewport={"width": 1920, "height": 1080})
        context.route("**/*", route_request)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(f"Normal motion: {error}"))
        page.goto(html.resolve().as_uri(), wait_until="load")
        page.keyboard.press("PageDown")
        page.keyboard.press("PageUp")
        assert_state((0, 0))
        real_motion = page.evaluate(REAL_MOTION_CHECK)
        assert not real_motion["failures"], real_motion["failures"]
        motion = page.evaluate("""async () => {
            const root = document.createElement('div'); document.body.append(root);
            const html = (x,r) => `<svg><g data-morph="probe" transform="translate(${x} 0)">` +
                `<circle r="${r}"/></g></svg>`;
            const position = () => Number(root.querySelector('g').getAttribute('transform').match(/[-.\\d]+/)[0]);
            MQSF_MOTION.replace(root,html(0,10),0);
            MQSF_MOTION.replace(root,html(100,20),1000);
            await new Promise(resolve=>setTimeout(resolve,200));
            const forward = position();
            MQSF_MOTION.replace(root,html(0,10),1000);
            const reverseStart = position();
            await new Promise(resolve=>setTimeout(resolve,200));
            const reverse = position();
            MQSF_MOTION.replace(root,html(250,30),1000);
            MQSF_MOTION.finish();
            const finished = root.querySelector('g').getAttribute('transform');
            for(let i=0;i<50;i++) MQSF_MOTION.replace(root,html(i*10,10+i),1000);
            MQSF_MOTION.replace(root,html(999,9),0);
            await new Promise(resolve=>setTimeout(resolve,100));
            const result = {forward, reverseStart, reverse, finished,
                transform:root.querySelector('g').getAttribute('transform'),
                radius:root.querySelector('circle').getAttribute('r'),
                animations:root.getAnimations({subtree:true}).length};
            root.remove(); return result;
        }""")
        forward = motion.pop("forward")
        assert 0 < forward < 100, "Forward morph must change geometry between endpoints"
        assert abs(motion.pop("reverseStart") - forward) < 0.001, "Reversing a morph must retain its current position"
        assert 0 < motion.pop("reverse") < 100, "Reverse morph must change geometry between endpoints"
        assert motion == {
            "finished": "translate(250 0)",
            "transform": "translate(999 0)",
            "radius": "9",
            "animations": 0,
        }
        # Leaving an active timeline must invalidate its pending frame callbacks.
        page.evaluate("index => { MQSF_DECK.go(index,0); MQSF_DECK.next(); MQSF_DECK.go(0); }", ids["routing"])
        page.wait_for_timeout(100)
        assert_state((0, 0))
        page.evaluate("MQSF_DECK.finishMotion()")
        page.emulate_media(reduced_motion="reduce")
        for index, slide in enumerate(slides):
            for step in map(int, slide["playbacks"]):
                page.evaluate("([index, step]) => MQSF_DECK.go(index, step, true)", [index, step])
                assert not assert_state((index, step))["animating"], "Reduced motion must settle without a timeline"
                assert not page.locator("#slide").get_attribute("data-progress")
                inspect_evidence()
                assert page.evaluate("document.querySelector('#slide').getAnimations({subtree:true}).length") == 0
        assert not errors, f"Browser errors: {errors}"
        assert not network, f"Presentation requested network resources: {network}"
        context.close()
        browser.close()
        for (index, issue), step in geometry.items():
            print(f"Layout: slide {index + 1}, build {step}: {issue}")
        assert not geometry, f"Found {len(geometry)} FullHD layout issues (listed above)"
        print(
            f"Offline {engine} checks passed: {len(slides)} slides, {len(states)} builds, "
            f"{completed} finite replays, {loops} loops."
        )
        print(
            f"Evidence: {probe['dependencies']} feedback dependencies, {probe['windows']} circuit windows, "
            f"exact result progress, {len(slides)} print pages."
        )
        print(
            f"Real RAF: {len(real_motion['samples'])} playback builds, {real_motion['frames']} sampled frames, "
            f"p95 frame interval {real_motion['p95']:.1f} ms; "
            "first paint, restart, cancellation, and reduced motion passed."
        )


def main() -> None:
    """Run the standalone browser check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--engine", choices=("chromium", "firefox"), default="chromium")
    parser.add_argument(
        "--browser",
        default=os.environ.get("MQSF_BROWSER"),
    )
    parser.add_argument("--screenshot", type=Path, help="Save the opening slide")
    parser.add_argument("--screenshots-dir", type=Path, help="Save the final build of every slide at FullHD")
    args = parser.parse_args()
    if not args.browser and args.engine == "chromium":
        args.browser = os.environ.get("CHROME_BIN") or (
            "/usr/bin/google-chrome" if Path("/usr/bin/google-chrome").is_file() else None
        )
    check(args.html, args.browser, args.screenshot, args.screenshots_dir, args.engine)


if __name__ == "__main__":
    main()
