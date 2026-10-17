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
    const overflowFor = (element, rect) => {
        const scale = bounds.width / 1760;
        const top = bounds.top - (element.closest('h1, h2') ? 18 * scale : 0);
        const footerMargin = element.closest('.opening-references') ? 23 :
            element.closest('.structure-footer') ? 20 : 0;
        const bottom = bounds.bottom + footerMargin * scale;
        return Math.max(bounds.left - rect.left, rect.right - bounds.right, top - rect.top, rect.bottom - bottom);
    };
    const visible = node => {
        if (node.closest('[hidden], [aria-hidden="true"], .reveal.off')) return false;
        for (let parent = node; parent && parent !== document.documentElement; parent = parent.parentElement) {
            const style = getComputedStyle(parent);
            if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) < 0.05)
                return false;
        }
        return node.getBoundingClientRect().width > 0;
    };
    for (const node of slide.querySelectorAll('h1, h2, h3, p, pre, img, svg, .metric, .code-frame')) {
        if (!visible(node) || node.parentElement.closest('svg')) continue;
        const rect = node.getBoundingClientRect();
        const name = node.tagName.toLowerCase() + (node.getAttribute('class') ? '.' +
            node.getAttribute('class').trim().replaceAll(' ', '.') : '');
        const label = (node.textContent || node.getAttribute('alt') || '').trim().slice(0, 70);
        const overflow = overflowFor(node, rect);
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
    const textRects = [], walker = document.createTreeWalker(slide, NodeFilter.SHOW_TEXT);
    const addText = (element, text, rect) => {
        let {left, right, top, bottom} = rect;
        for (let parent = element.parentElement; parent && parent !== slide; parent = parent.parentElement) {
            const style = getComputedStyle(parent), clip = parent.getBoundingClientRect();
            if (style.overflowX !== 'visible') {
                left = Math.max(left, clip.left); right = Math.min(right, clip.right);
            }
            if (style.overflowY !== 'visible') {
                top = Math.max(top, clip.top); bottom = Math.min(bottom, clip.bottom);
            }
        }
        if (text.trim() && right - left > 1 && bottom - top > 1)
            textRects.push({element, text:text.trim().slice(0, 65), rect:{left, right, top, bottom}});
    };
    let node;
    while ((node = walker.nextNode())) {
        if (node.parentElement.closest('svg, script, style') || !visible(node.parentElement)) continue;
        const range = document.createRange();
        range.selectNodeContents(node);
        for (const rect of range.getClientRects()) addText(node.parentElement, node.textContent, rect);
    }
    for (const text of slide.querySelectorAll('svg text'))
        if (visible(text)) addText(text, text.textContent, text.getBoundingClientRect());
    for (const {element, text, rect} of textRects) {
        const overflow = overflowFor(element, rect);
        if (overflow > 3) issues.push(`text: outside slide by ${Math.round(overflow)}px: ${text}`);
    }
    for (let i = 0; i < textRects.length; i++) {
        for (let j = i + 1; j < textRects.length; j++) {
            const a = textRects[i], b = textRects[j];
            if (a.element === b.element) continue;
            const width = Math.min(a.rect.right, b.rect.right) - Math.max(a.rect.left, b.rect.left);
            const height = Math.min(a.rect.bottom, b.rect.bottom) - Math.max(a.rect.top, b.rect.top);
            if (width > 3 && height > 3)
                issues.push(`text overlap (${Math.round(width)}x${Math.round(height)}px): ${a.text} / ${b.text}`);
        }
    }
    return issues;
}"""


EVIDENCE_CHECK = """() => {
    const id = document.querySelector('.batch-grid') ? 'afqmc-execution' :
        document.querySelector('.histogram') ? 'adaptive-execution' : '';

    let capture, actual, expected;
    const elapsed = Number(document.querySelector('.batch-grid, .histogram')?.dataset.elapsedMs);
    if (id === 'adaptive-execution') {
        capture = MQSF_DATA.scenarios.find(s => s.id === 'qpe').variants[0].executions['qir-adaptive'];
        const shots = capture.shot_events.filter(s => s.time_ms <= elapsed);
        const counts = Array(18).fill(0);
        shots.forEach(s => {const n = parseInt(s.outcome, 2); counts[n < 78 ? 0 : n > 93 ? 17 : n - 77]++;});
        const histogram = document.querySelector('.histogram');
        actual = {count:Number(histogram.dataset.shots), bins:[...histogram.querySelectorAll('[data-count]')]
            .map(bar => Number(bar.dataset.count))};
        expected = {count:shots.length, bins:counts};
        const workers = [...new Set(capture.worker_events.map(e => e.worker_index))].sort((a,b) => a-b);
        expected.workers = workers.map(id => ({id, completed:capture.worker_events
            .findLast(e => e.worker_index === id && e.time_ms <= elapsed)?.completed || 0}));
        actual.workers = [...document.querySelectorAll('[data-worker]')]
            .map(e => ({id:Number(e.dataset.worker), completed:Number(e.dataset.completed)}));
    } else if (id === 'afqmc-execution') {
        capture = MQSF_DATA.application.execution;
        const grid = document.querySelector('.batch-grid');
        const complete = rows => rows.filter(e => e.time_ms + (e.duration_ms || 0) <= elapsed).length;
        expected = {compiled:complete(capture.compilation), completed:complete(capture.program_completions),
            count:complete(capture.retrievals)};
        actual = {compiled:Number(grid.dataset.compiled), completed:Number(grid.dataset.completed),
            count:Number(grid.dataset.programs)};
        expected.cells = [capture.num_programs - expected.compiled, expected.compiled - expected.completed,
            expected.completed - expected.count, expected.count];
        if (grid.classList.contains('preparation-grid')) {
            expected.prepared = complete(capture.preparation);
            actual.prepared = Number(grid.dataset.prepared);
            expected.cells = [capture.num_programs - expected.prepared, 0, 0, expected.prepared];
        }
        actual.cells = [...grid.querySelectorAll('svg > path')]
            .map(p => (p.getAttribute('d').match(/M/g) || []).length);
        if (MQSF_DECK.getState().step >= 3) {
            const application = MQSF_DATA.application, propagation = application.propagation;
            const clock = Number(document.querySelector('.workflow-clock').dataset.elapsedMs);
            const wallTime = clock - application.timeline.propagation_started_ms;
            const panel = document.querySelector('.app-panel:nth-child(3)');
            expected.energyPoints = propagation.curves.map(c => c.step_completed_ms.filter(t => t <= wallTime).length);
            actual.energyPoints = [...panel.querySelectorAll('[data-points]')].map(e => Number(e.dataset.points));
            expected.weights = propagation.curves.flatMap(curve => {
                const times = curve.frames.map(frame => Math.max(...frame.walkers.map(w => w.completed_ms)));
                const next = times.findIndex(time => time > wallTime);
                const position = wallTime <= times[0] ? 0 : next < 0 ? times.length - 1 :
                    next - 1 + (wallTime - times[next - 1]) / (times[next] - times[next - 1]);
                const low = Math.floor(position), high = Math.min(low + 1, times.length - 1);
                return curve.frames[low].walkers.map((walker, i) => walker.weight +
                    (curve.frames[high].walkers[i].weight - walker.weight) * (position - low));
            });
            const walkers = [...panel.querySelectorAll('[data-weight]')];
            actual.weights = walkers.map(e => Number(e.dataset.weight));
            expected.identities = propagation.curves.flatMap((curve, trial) =>
                curve.frames[0].walkers.map(w => `${trial}:${w.id}`));
            actual.identities = walkers.map(e => `${e.dataset.trial}:${e.dataset.walkerId}`);
            expected.visibleRings = expected.identities.length;
            actual.visibleRings = walkers.filter(e =>
                Number(e.querySelector('circle').getAttribute('r')) === 14).length;
        }
    } else return null;
    const state = MQSF_DECK.getState(), progress = document.querySelector('#slide').dataset.progress;
    const active = id === 'afqmc-execution' ? [1, 2, 3].includes(state.step) : state.step === 1;
    const fraction = active && progress !== '' ? Number(progress) : 1;
    return {actual, expected, fraction};
}"""

CIRCUIT_CHECK = """() => {
    const {schedule, render, flatten} = MQSF_CIRCUIT, failures = [];
    const qpe = MQSF_DATA.scenarios.find(s => s.id === 'qpe').variants[0].stages.find(s => s.id === 'source').circuit;
    const qpeView = document.createElement('div');
    qpeView.innerHTML = MQSF_CIRCUIT.renderQPE(qpe);
    const capturedGates = flatten(qpe).filter(op => op.qubits.length).map(op => `${op.name}:${op.source_line}`);
    const shownGates = [...qpeView.querySelectorAll('[data-operation]')].map(op =>
        `${op.dataset.operation}:${op.dataset.sourceLine}`);
    if (shownGates.join('|') !== capturedGates.join('|') ||
        qpeView.querySelectorAll('[data-control-flow="for_loop"]').length !== 2)
        failures.push('The complete QPE diagram loses source gates or one of its nested loops');
    for (const mutate of [
        c => c.operations.push({name:'h', qubits:[1]}),
        c => c.operations[0].qubits = [2],
        c => c.operations.at(-1).blocks[0][0].qubits = [1],
        c => c.operations.at(-1).blocks[0][3].qubits = [1],
        c => c.operations.at(-1).blocks[0][5].qubits = [1],
    ]) {
        const altered = structuredClone(qpe); mutate(altered);
        let rejected = false;
        try { MQSF_CIRCUIT.renderQPE(altered); } catch { rejected = true; }
        if (!rejected) failures.push('The QPE diagram silently substitutes the original program after a source change');
    }
    const parity = MQSF_DATA.scenarios.find(s => s.id === 'parity').variants[0];
    const source = parity.stages.find(s => s.id === 'source').circuit;
    const h = schedule(source).filter(o => o.name === 'h');
    if (h.length < 2 || h[0].column !== h[1].column) failures.push('Independent parity H gates are not parallel');
    const feedback = document.createElement('div');
    feedback.innerHTML = render(source, {limit:20});
    if (!feedback.querySelector('[data-morph*="-classical-"]')) failures.push('Missing measured-bit feedback wire');
    feedback.innerHTML = render(source, {full:true, fold:2});
    if (feedback.querySelector('.circuit').dataset.foldRows !== '1' ||
        !feedback.querySelector('[data-morph*="-classical-"]'))
        failures.push('Folding must preserve structured scopes and feedback');
    const rus = MQSF_DATA.scenarios.find(s => s.id === 'rus').variants.find(v => v.id === 'structured');
    const rusCircuit = rus.stages.find(s => s.id === 'source').circuit;
    const correction = flatten(rusCircuit).find(o => o.name === 'z' &&
        o.region.filter(r => r.name === 'if_else').length === 2);
    if (!correction) failures.push('RUS example lost its nested correction guards');
    else {
        feedback.innerHTML = render(rusCircuit, {limit:100});
        const wires = [...feedback.querySelectorAll(`[data-feedback-to="${correction.id}"]`)];
        if (wires.length !== 2 || new Set(wires.map(w => w.dataset.conditionBit)).size !== 2)
            failures.push('RUS correction must receive feedback from both measured guard bits');
        if (/=\\s*1|if\\s+1/.test(feedback.textContent))
            failures.push('Circuit invents a positive predicate for the negated RUS guard');
        const finalH = flatten(rusCircuit).findLast(o => o.name === 'h' && !o.region.length);
        const finalGate = feedback.querySelector(`[data-morph="circuit-op-${finalH.id}"]`);
        const finalX = Number(finalGate.getAttribute('transform').match(/translate\\(([-.\\d]+)/)[1]);
        const loop = feedback.querySelector('[data-morph^="circuit-region-"] rect');
        if (Number(loop.getAttribute('x')) + Number(loop.getAttribute('width')) >= finalX - 25)
            failures.push('RUS loop boundary includes its final post-loop basis rotation');
    }
    let dependencies = 0, windows = 0;
    function checkWindow(c, options, label) {
        const holder = document.createElement('div');
        holder.innerHTML = render(c, options);
        const active = Array.isArray(options.active) ? options.active : [options.active];
        if (holder.querySelectorAll('.gate-highlight').length !== active.length)
            failures.push(`${label}: active gates missing`);
        const wire = holder.querySelector('.circuit').dataset;
        for (const gate of holder.querySelectorAll('[data-operation]')) {
            const x = Number(gate.getAttribute('transform').match(/translate\\(([-.\\d]+)/)[1]);
            if (x < Number(wire.left) || x > Number(wire.right)) failures.push(`${label}: gate outside wires`);
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
            if (stage.id === 'target-native-synthesis') {
                const holder = document.createElement('div'), scheduled = schedule(c);
                holder.innerHTML = render(c, {full:true, fold:20});
                const gates = [...holder.querySelectorAll('[data-operation]')];
                if (gates.length !== scheduled.length || gates.some((gate,i) =>
                    gate.dataset.operation !== scheduled[i].name ||
                    Number(gate.dataset.column) !== scheduled[i].column ||
                    Number(gate.dataset.circuitRow) !== Math.floor(scheduled[i].column / 20)))
                    failures.push(`${target.id}: folding changes the program or its dependency order`);
                const rows = Number(holder.querySelector('.circuit').dataset.foldRows);
                if (holder.querySelectorAll('[data-morph*="-wire-"]').length !== rows * c.qubits.length)
                    failures.push(`${target.id}: folded circuit loses a wire`);
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
    const applicationIndex = MQSF_DECK.slides.findIndex(s => s.id === 'afqmc-execution');
    MQSF_DECK.go(applicationIndex, MQSF_DECK.slides[applicationIndex].builds);
    MQSF_DECK.finishMotion();
    MQSF_DECK.next();
    const chapter = document.querySelector('.chapter-separator');
    if (!chapter) failures.push('The compiler chapter did not follow the AFQMC application');
    else {
        if (Number(getComputedStyle(chapter).opacity) !== 0)
            failures.push('The new compiler chapter flashed before its entrance');
        await waitFrames(140);
        const opacity = Number(getComputedStyle(chapter).opacity);
        if (!(opacity > 0 && opacity < 1))
            failures.push('The compiler chapter did not fade through an intermediate frame');
        MQSF_DECK.finishMotion();
        if (Number(getComputedStyle(chapter).opacity) !== 1)
            failures.push('The compiler chapter did not finish fully visible');
    }
    const targetIndex = MQSF_DECK.slides.findIndex(s => s.id === 'routing');
    for (const [source, target] of [[4,5],[5,6]]) {
        const selector = '.stage-place-and-route .circuit';
        const rectangle = () => {
            const r = document.querySelector(selector).getBoundingClientRect();
            return [r.x,r.y,r.width,r.height];
        };
        MQSF_DECK.go(targetIndex, target); MQSF_DECK.finishMotion();
        const destination = rectangle();
        MQSF_DECK.go(targetIndex, source); MQSF_DECK.finishMotion();
        const origin = rectangle();
        MQSF_DECK.next();
        if (rectangle().some((v,i) => Math.abs(v-origin[i]) > .01))
            failures.push(`Compiler card ${source} -> ${target} jumps at shrink start`);
        await waitFrames(350);
        const dimension = origin[2] !== destination[2] ? 2 : 3, midway = rectangle()[dimension];
        if (!(midway > destination[dimension] && midway < origin[dimension]))
            failures.push(`Compiler card ${source} -> ${target} does not shrink continuously`);
        MQSF_DECK.finishMotion();
        if (rectangle().some((v,i) => Math.abs(v-destination[i]) > .01))
            failures.push(`Compiler card ${target} does not settle at its target size`);
    }
    const positions = () => new Map([...document.querySelectorAll('[data-morph^="device-node-"]')]
        .map(node => {
            // consolidate() rewrites Firefox's attribute and changes the geometry under test.
            const {e, f} = node.transform.baseVal.getItem(0).matrix;
            return [node.dataset.morph, [e, f]];
        }));
    for (const [source, target] of [[6,7],[7,8]]) {
        MQSF_DECK.go(targetIndex, target); MQSF_DECK.finishMotion();
        const destination = positions();
        MQSF_DECK.go(targetIndex, source); MQSF_DECK.finishMotion();
        const origin = positions();
        const key = [...origin.keys()].find(k => destination.has(k)
            && Math.hypot(...destination.get(k).map((v, i) => v - origin.get(k)[i])) > 5);
        if (!key) { failures.push('No shared target node to test'); continue; }
        const from = origin.get(key), to = destination.get(key);
        const distance = point => Math.hypot(...point.map((v, i) => v - from[i]));
        MQSF_DECK.next();
        if (MQSF_DECK.getState().animating || document.querySelector('#slide').dataset.progress !== '')
            failures.push(`Target ${target} uses a staged replay instead of one morph`);
        if (distance(positions().get(key)) > .001)
            failures.push(`Target ${target} flashed its final topology before morphing`);
        await waitFrames(350);
        const fraction = distance(positions().get(key)) / distance(to);
        if (!(fraction > 0 && fraction < 1))
            failures.push(`Target ${target} did not morph through an intermediate position: ${fraction}`);
        MQSF_DECK.finishMotion();
        if (Math.abs(distance(positions().get(key)) - distance(to)) > .001)
            failures.push(`Target ${target} did not settle its topology on finish`);
        MQSF_DECK.go(targetIndex, source); MQSF_DECK.finishMotion();
        MQSF_DECK.go(targetIndex, target);
        await new Promise(requestAnimationFrame);
        MQSF_DECK.go(0);
        await waitFrames(1000);
        if (MQSF_DECK.getState().slide !== 0 || document.querySelector('#slide').dataset.progress !== '')
            failures.push(`Target ${target} transition survived navigation`);
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
            // One click must cancel the running replay and advance immediately.
            MQSF_DECK.next();
            const expected = step < slide.builds ? [index, step + 1] : [index + 1, 0];
            const advanced = MQSF_DECK.getState();
            if (advanced.slide !== expected[0] || advanced.step !== expected[1])
                failures.push(`${slide.id}/${step}: click did not advance the running replay`);
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


def afqmc_curve_frontier(page: Any) -> None:  # ruff: ignore[any-type] - Optional Playwright import.
    """Interpolating the visible curve must never draw backwards along imaginary time."""
    before = page.evaluate("MQSF_DECK.getState()")
    index = page.evaluate("MQSF_DECK.slides.findIndex(s => s.id === 'hybrid-algorithm')")
    playback = page.evaluate("MQSF_DECK.slides.find(s => s.id === 'hybrid-algorithm').playbacks[2]")
    page.evaluate("i => MQSF_DECK.go(i, 2, true)", index)
    page.clock.run_for(playback["transition"])
    previous = [0.0, 0.0]
    for _ in range(101):
        curves = page.locator(".energy-chart polyline[data-points]").evaluate_all("""elements => elements.map(e =>
            Array.from({length:e.points.numberOfItems}, (_,i) => e.points.getItem(i).x))""")
        assert len(curves) == 2
        for curve, frontier in zip(curves, previous, strict=True):
            assert curve
            assert curve == sorted(curve), f"AFQMC curve backtracks: {curve}"
            assert curve[-1] >= frontier, "AFQMC curve frontier moved backwards"
        previous = [curve[-1] for curve in curves]
        page.clock.fast_forward(round(playback["duration"] / 100))
    page.evaluate("MQSF_DECK.finishAnimation(); MQSF_DECK.finishMotion()")
    page.evaluate("s => { MQSF_DECK.go(s.slide, s.step); MQSF_DECK.finishMotion(); }", before)


def compiler_semantics(page: Any) -> None:  # ruff: ignore[any-type] - Optional Playwright import.
    """Check exact compiler excerpts, physical wire identity, and placement reveal."""
    failures = page.evaluate(r"""() => {
        const failures = [], holder = document.createElement('div');
        const before = MQSF_DECK.getState();
        const render = (id, step, initial = false) => {
            MQSF_DECK.go(MQSF_DECK.slides.findIndex(s => s.id === id), step, initial);
            MQSF_DECK.finishMotion();
            const html = document.querySelector('#slide').innerHTML;
            MQSF_DECK.finishAnimation();
            return html;
        };
        const stages = t => [...t.compilation.stages, ...t.compilation.exports];
        const source = MQSF_DATA.targets[0];
        const circuit = {qubits:[{id:0, site:17, label:'$17'}, {id:1, site:18, label:'$18'}],
            operations:[{id:'cx', name:'cx', qubits:[1,0]}]};
        holder.innerHTML = MQSF_CIRCUIT.render(circuit, {full:true, layout:[18,17]});
        if (holder.querySelector('svg').viewBox.baseVal.width !== 225)
            failures.push('Short circuit retains trailing idle whitespace');
        const labels = [...holder.querySelectorAll('[data-morph*="-wire-"] text')].map(e => e.textContent);
        if (labels.join('|') !== 'q1 ↦ 17|q0 ↦ 18') failures.push('Physical wire labels invert placement incorrectly');
        holder.innerHTML = render('routing', 1);
        if (!holder.querySelector('.stage-optimized') || holder.querySelector('.spec-card'))
            failures.push('Device-independent optimization must precede device selection');
        holder.innerHTML = render('routing', 2);
        if (!holder.querySelector('.spec-step').textContent.includes('3 · Device Specification') ||
            !holder.querySelector('.stage-optimized h3').textContent.includes('2 · Optimization'))
            failures.push('Compiler stage numbering differs between tiles and cards');
        const tiles = [...holder.querySelectorAll('.compile-tile b')].map(e => e.textContent);
        if (tiles[1] !== 'Optimization' || tiles[2] !== 'Device Specification')
            failures.push('Compiler tile order differs from its reveal order');
        const initial = render('routing', 3), replayStart = render('routing', 4, true);
        holder.innerHTML = initial;
        const initialWires = [...holder.querySelectorAll('.stage-place-and-route [data-morph*="-wire-"]')]
            .map(e => e.outerHTML).join('');
        for (const [label, html] of [['placement',initial], ['replay start',replayStart]]) {
            holder.innerHTML = html;
            if (holder.querySelector('.stage-place-and-route .program-metrics'))
                failures.push(label + ' reveals routing metrics before routing completes');
            if ([...holder.querySelectorAll('.stage-place-and-route [data-operation]')]
                .some(e => Number(e.getAttribute('opacity')) !== 0)) failures.push(label + ' reveals operations');
            if ([...holder.querySelectorAll('.stage-place-and-route [data-morph*="-wire-"]')]
                .map(e => e.outerHTML).join('') !== initialWires)
                failures.push('Routing replay starts from another layout');
        }
        holder.innerHTML = render('routing', 4);
        if (!holder.querySelector('.stage-place-and-route .program-metrics'))
            failures.push('Completed routing has no metrics');
        for (const [index, step] of [[0,6],[1,7],[2,8]]) {
            const target = MQSF_DATA.targets[index];
            holder.innerHTML = render('routing', step);
            for (const id of ['source','optimized']) {
                const card = holder.querySelector('.stage-' + id);
                if (card.dataset.target !== source.id || !card.querySelector('.program-metrics'))
                    failures.push('Target switch changes an invariant compiler stage');
            }
            for (const metrics of holder.querySelectorAll('.program-metrics')) {
                const labels = [...metrics.querySelectorAll('span')].map(e => e.textContent);
                if (labels.join('|') !== '1Q|2Q|D')
                    failures.push('Compiler metrics must retain the three compact rows');
            }
            for (const id of ['place-and-route','target-native-synthesis']) {
                const card = holder.querySelector('.stage-' + id);
                if (card.dataset.target !== target.id) failures.push('Target switch retains a stale stage');
                const rows = Number(card.querySelector('.circuit').dataset.foldRows);
                const actual = [...card.querySelectorAll('[data-morph*="-wire-"]')]
                    .map(e => e.querySelector('text').textContent);
                const wires = [...new Set(stages(target).find(s => s.id === id).circuit.qubits.map(q => q.site))]
                    .sort((a,b) => a-b).map(site => `q${target.compilation.layout.initial.indexOf(site)} ↦ ${site}`);
                const expected = Array.from({length:rows}, () => wires).flat();
                if (actual.join('|') !== expected.join('|'))
                    failures.push('Native stage loses initial placement labels');
                if (id === 'target-native-synthesis') {
                    const count = MQSF_CIRCUIT.flatten(stages(target).find(s => s.id === id).circuit).length;
                    if (card.querySelectorAll('[data-operation]').length !== count || rows < 2)
                        failures.push('Folded native circuit drops captured operations');
                }
            }
            const format = ['qir-adaptive','qpy','openqasm3'][index];
            const artifact = stages(target).find(s => s.id === format);
            const lines = artifact.code.split('\n');
            const output = holder.querySelector('.compiler-output');
            const actual = [...output.querySelectorAll('.code-text')].map(e => e.textContent);
            const first = index === 0 ? lines.findIndex(line => line.includes('define')) : 0;
            const count = [13,5,16][index];
            const expected = lines.slice(first,first+count);
            if (output.dataset.format !== format || actual.join('\n') !== expected.join('\n'))
                failures.push('Executable format or excerpt differs from its captured target export');
            if (format === 'qpy' && output.querySelector('.qiskit-output')?.getAttribute('src') !==
                'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(artifact.svg))
                failures.push('IBM executable does not use its captured Qiskit drawing');
        }
        const parity = MQSF_DATA.scenarios.find(s => s.id === 'parity').variants.find(v => v.id === 'structured');
        const qir = parity.exports.find(s => s.id === 'qir-adaptive').code.split('\n');
        const last = qir.findIndex((line,i) => line === '}' && qir[i-1]?.includes('ret i64 0'));
        holder.innerHTML = render('structured', 1);
        const included = new Set();
        for (const line of holder.querySelectorAll('.structured-qir .full-code .code-line')) {
            if (line.classList.contains('omitted-native')) {
                failures.push('Portable QIR overview must not omit gate calls');
            } else {
                const number = Number(line.querySelector('.line-no').textContent);
                included.add(number);
                if (line.querySelector('.code-text').textContent !== qir[number-1])
                    failures.push('Portable QIR overview changes a captured source line');
            }
        }
        if (last < 0 || qir.slice(0,last+1).some((line,i) => line.trim() && !included.has(i+1)) ||
            [...included].some(n => n > last+1)) failures.push('Portable QIR overview loses entry-point structure');
        MQSF_DECK.go(before.slide, before.step);
        MQSF_DECK.finishMotion();
        return failures;
    }""")
    assert not failures, "\n".join(failures)


STABILITY_CHECK = """async () => {
    const failures = [], samples = [], ids = Object.fromEntries(MQSF_DECK.slides.map((s,i) => [s.id,i]));
    const wait = ms => new Promise(resolve => setTimeout(resolve,ms));
    const require = (condition, message) => { if (!condition) failures.push(message); };
    const opacity = node => {
        let value = 1;
        for (; node && node !== document.body; node = node.parentElement)
            value *= Number(getComputedStyle(node).opacity);
        return value;
    };
    // Adjacent builds retain the exact decoded reference images, including nested keys.
    MQSF_DECK.go(ids['hybrid-algorithm'],0); MQSF_DECK.finishMotion();
    const reference = document.querySelector('[data-morph="application-reference"]');
    const qr = reference.querySelector('[data-morph="qr-afqmc"] img');
    const paper = document.querySelector('[data-morph="application-paper-reference"]');
    const paperQr = paper.querySelector('img');
    for (const step of [1,2]) {
        MQSF_DECK.go(ids['hybrid-algorithm'],step);
        for (const delay of [0,450]) {
            if (delay) await wait(delay);
            require(document.querySelector('[data-morph="application-reference"]') === reference,
                `AFQMC build ${step}: replaced unchanged reference block`);
            require(document.querySelector('[data-morph="qr-afqmc"] img') === qr,
                `AFQMC build ${step}: replaced unchanged QR image`);
            require(opacity(qr) > .999, `AFQMC build ${step}: reference faded`);
            require(document.querySelector('[data-morph="application-paper-reference"]') === paper &&
                paper.querySelector('img') === paperQr && opacity(paperQr) > .999,
                `AFQMC build ${step}: paper reference changed or faded`);
        }
        MQSF_DECK.finishMotion();
        samples.push(`reference-${step}`);
    }
    // Updating a recorded time must not fade the surrounding clock or an ancestor.
    for (const [id,key] of [['afqmc-execution','workflow-clock'],['adaptive-execution','runtime-clock']]) {
        MQSF_DECK.go(ids[id],0); MQSF_DECK.finishMotion();
        const selector = `[data-morph="${key}"]`, before = document.querySelector(selector).textContent;
        MQSF_DECK.go(ids[id],1);
        const clock = document.querySelector(selector);
        require(clock.textContent !== before, `${id}: probe did not advance the recorded clock`);
        require(opacity(clock) > .999, `${id}: clock faded at transition start`);
        await wait(450);
        require(opacity(clock) > .999, `${id}: clock faded at transition midpoint`);
        MQSF_DECK.finishMotion();
        samples.push(key);
    }
    const root = document.createElement('div');
    root.style.cssText = 'position:fixed;left:40px;top:40px;width:800px;height:140px';
    document.body.append(root);
    const nested = (x, child) => `<div data-morph="stability-parent" style="position:absolute;` +
        `left:${x}px;top:0;width:200px;height:80px"><span data-morph="stability-child" ` +
        `style="position:absolute;left:${child}px;top:20px;width:40px;height:20px">child</span></div>`;
    const node = key => root.querySelector(`[data-morph="stability-${key}"]`);
    MQSF_MOTION.replace(root,nested(0,20),0);
    const start = node('child').getBoundingClientRect().x;
    MQSF_MOTION.replace(root,nested(300,20),1000);
    require(node('parent').getAnimations().length === 1, 'Moving parent has no FLIP animation');
    require(node('child').getAnimations().length === 0, 'Nested child received a second parent FLIP');
    require(Math.abs(node('child').getBoundingClientRect().x - start) < 1,
        'Nested child jumped at transition start');
    const parentAnimation = node('parent').getAnimations()[0];
    parentAnimation?.pause();
    if (parentAnimation) parentAnimation.currentTime = 500;
    const midpoint = node('child').getBoundingClientRect().x;
    require(midpoint > start && midpoint < start + 300, 'Nested motion did not interpolate between endpoints');
    require(Math.abs(node('child').getBoundingClientRect().x - node('parent').getBoundingClientRect().x - 20) < 1,
        'Nested child moved twice relative to its parent');
    MQSF_MOTION.finish();
    // An unchanged outer box must still permit motion within that box.
    const innerStart = node('child').getBoundingClientRect().x;
    MQSF_MOTION.replace(root,nested(300,140),1000);
    require(node('parent').getAnimations().length === 0, 'Stationary parent received a FLIP animation');
    require(node('child').getAnimations().length === 1, 'Stationary parent suppressed child layout motion');
    require(Math.abs(node('child').getBoundingClientRect().x - innerStart) < 1,
        'Child layout change jumped at transition start');
    MQSF_MOTION.finish(); root.remove();
    samples.push('nested-motion');
    const footer = document.querySelector('#deck > .slide-footer > a'), centers = [];
    for (let slide = 0; slide < MQSF_DECK.slides.length; slide++) {
        MQSF_DECK.go(slide,0); MQSF_DECK.finishMotion();
        const rect = footer.getBoundingClientRect();
        centers.push((rect.left + rect.right) / 2);
        require(document.querySelector('#deck > .slide-footer > a') === footer, 'Recreated persistent website footer');
    }
    require(Math.max(...centers) - Math.min(...centers) < .1, 'Website footer moved with slide/build counts');
    require(centers.every(x => Math.abs(x - innerWidth / 2) < 1), 'Website footer is not centered on the projector');
    samples.push('persistent-footer');
    MQSF_DECK.go(0); MQSF_DECK.finishMotion();
    return {failures,samples};
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
        assert len(slides) == 17
        assert len({slide["id"] for slide in slides}) == len(slides), "Slide IDs must be unique"
        states = [(index, step) for index, slide in enumerate(slides) for step in range(slide["builds"] + 1)]
        ids = {slide["id"]: index for index, slide in enumerate(slides)}
        opening = page.evaluate("""() => {
            const holder = document.createElement('div');
            const routes = p => {
                holder.innerHTML = MQSF_VIZ.architecture('problem', p);
                return [...holder.querySelectorAll('path[data-morph^="world-"], path[data-morph="hpc-cloud-burst"]')]
                    .filter(n => Number(n.getAttribute('opacity')) > 0)
                    .map(n => n.dataset.morph);
            };
            const keys = [];
            const frontends = p => {
                holder.innerHTML = MQSF_VIZ.architecture('stack', p);
                const visible = selector => [...holder.querySelectorAll(selector)]
                    .filter(n => Number(n.getAttribute('opacity')) > 0).length;
                return {
                    cards: visible('g[data-morph^="frontend-"]:not([data-morph="frontend-split"])'),
                    routes: visible('path[data-morph^="user-frontend-"], path[data-morph^="user-other-frontend-"]'),
                };
            };
            for (const [mode, phases] of Object.entries(MQSF_VIZ.architecturePhases))
                for (const phase of phases.keys()) {
                    holder.innerHTML = MQSF_VIZ.architecture(mode, phase);
                    keys.push([...holder.querySelectorAll('[data-morph]')].map(n => n.dataset.morph));
                }
            return {start:routes(0), first:routes(1), complete:routes(6), keys,
                beforeFrontends:frontends(0), afterFrontends:frontends(1)};
        }""")
        assert not opening["start"], "Future deployment paths must not flash on entry"
        assert opening["first"] == ["world-user-compute-0-0", "world-compute-qpu-0-0"]
        assert set(opening["complete"]) == {
            *(f"world-user-compute-{u}-{c}" for u in range(4) for c in range(3)),
            *(f"world-compute-qpu-{c}-{q}" for c in range(3) for q in range(5)),
            "hpc-cloud-burst",
        }, "Every deployment route must pass through classical compute"
        assert opening["beforeFrontends"] == {"cards": 0, "routes": 0}
        assert opening["afterFrontends"] == {"cards": 4, "routes": 16}
        assert len(opening["keys"]) == 16
        expected_keys = set(opening["keys"][0])
        for _phase, keys in enumerate(opening["keys"]):
            additions = set()
            assert len(keys) == len(set(keys)), "Architecture morph keys must be unique"
            assert set(keys) == expected_keys | additions, (
                "Every architecture phase must retain its shared, unique morph keys"
            )
        compiler_semantics(page)
        page.evaluate("MQSF_DECK.go(MQSF_DECK.slides.findIndex(s => s.id === 'features'), 4)")
        assert page.locator(".feature-survey").evaluate("panel => panel.inert")
        page.evaluate("MQSF_DECK.next(); MQSF_DECK.finishMotion()")

        def survey_frame() -> dict[str, Any]:
            return page.evaluate("""() => {
                const panel = document.querySelector('.feature-survey');
                return {
                    panel:Number(panel.style.opacity),
                    lastCard:Number(document.querySelector('[data-morph="feature-5"]').style.opacity),
                    refs:Number(document.querySelector('.feature-resources').style.opacity),
                    inert:panel.inert,
                };
            }""")

        assert survey_frame() == {"panel": 0, "lastCard": 0, "refs": 1, "inert": True}
        page.clock.run_for(700)
        frame = survey_frame()
        assert frame["lastCard"] == 1
        assert frame["panel"] == 0
        assert frame["inert"]
        page.clock.run_for(500)
        frame = survey_frame()
        assert frame["lastCard"] == 1
        assert 0 < frame["panel"] < 1
        assert frame["refs"] == 0
        assert not frame["inert"]
        page.evaluate("MQSF_DECK.finishAnimation()")
        assert survey_frame() == {"panel": 1, "lastCard": 1, "refs": 0, "inert": False}
        assert (
            page.locator(".feature-survey a").evaluate_all("links => links.map(a => a.href)")
            == ["https://www.surveymonkey.com/r/QOSSSurvey26"] * 2
        )
        afqmc_curve_frontier(page)
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
            return evidence

        def inspect_slide(index: int, step: int) -> None:
            page.evaluate("MQSF_DECK.finishMotion()")
            duplicate_keys = page.evaluate("""() => {
                const keys = [...document.querySelectorAll('#slide [data-morph]')].map(n => n.dataset.morph);
                return [...new Set(keys.filter((key, i) => keys.indexOf(key) !== i))];
            }""")
            assert not duplicate_keys, f"Slide {index + 1} build {step} has duplicate motion keys: {duplicate_keys}"
            for issue in page.evaluate(LAYOUT_CHECK):
                geometry.setdefault((index, issue), step)
            assert page.locator("#slide-number").inner_text().startswith(f"{index + 1} / {len(slides)}")
            assert page.locator("#section-label").inner_text().strip()
            assert page.locator("#mqsc-logo").is_visible()
            inspect_evidence()
            if slides[index]["id"] == "structured":
                for artifact, container in (("source", ".structured-source"),):
                    source = page.evaluate(
                        """([scenario, id]) => {
                        const v = MQSF_DATA.scenarios.find(s => s.id === scenario).variants
                            .find(v => v.id === 'structured');
                        return [...v.stages, ...v.exports].find(a => a.id === id).code;
                    }""",
                        ["parity", artifact],
                    )
                    visible = {
                        line.strip() for line in page.locator(f"{container} .full-code .code-text").all_text_contents()
                    }
                    required = {line.strip() for line in source.splitlines() if line.strip()}
                    assert required
                    missing = sorted(required - visible)
                    assert not missing, f"Structured build {step} omits {artifact} lines: {missing}"
                assert page.locator(".structured-qir").evaluate(
                    "(node, hidden) => node.classList.contains('future') === hidden", step == 0
                ), "The source enters before its complete QIR counterpart"
            if slides[index]["id"] == "routing" and step in {3, 4, 5, 6, 7, 8}:
                target = 1 if step == 7 else 2 if step == 8 else 0
                mapping = page.evaluate(
                    """([target, initial]) => {
                    const c = MQSF_DATA.targets[target].compilation;
                    return initial ? c.layout.initial : c.layout.final.slice(0, c.layout.initial.length);
                }""",
                    [target, step == 3],
                )
                for i, site in enumerate(mapping):
                    marker = page.locator(f'[data-morph="logical-qubit-{i}"]')
                    physical = page.locator(f'[data-morph="device-node-{site}"]')
                    assert marker.get_attribute("transform") == physical.get_attribute("transform"), (
                        f"Target {target}, build {step}: logical qubit {i} must match captured site {site}"
                    )
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
                if (document.querySelector('.batch-grid')) return [.1, .4, .8, .95];
                const c = MQSF_DATA.scenarios.find(s => s.id === 'qpe').variants[0].executions['qir-adaptive'];
                const end = Math.max(0, c.completed_ms || 0, c.duration_ms || 0,
                    ...c.events.map(e => e.time_ms + (e.duration_ms || 0)),
                    ...(c.lifecycle || []).map(e => e.time_ms));
                const quantile = (rows, f) => rows.map(e => e.time_ms + (e.duration_ms || 0))
                    .sort((a,b) => a-b)[Math.floor(rows.length * f)];
                const times = [0.25, 0.5, 0.75].map(f => quantile(c.shot_events, f));
                const begin = c.submission_started_ms ??
                    c.events.find(e => /try_submit_job|job_submit/.test(e.operation))?.time_ms ?? 0;
                return times.sort((a,b) => a-b).map(t => (t - begin) / (end - begin));
            }""",
            )
            last = 0.0
            observations = []
            for fraction in fractions:
                page.clock.fast_forward(round(playback["duration"] * (fraction - last)))
                evidence = inspect_evidence()
                assert evidence is not None
                observations.append(evidence["actual"])
                for issue in page.evaluate(LAYOUT_CHECK):
                    geometry.setdefault((index, issue), step)
                last = fraction
            if "prepared" in observations[0]:
                assert 0 <= observations[0]["prepared"] < observations[-1]["prepared"], observations
            elif "energyPoints" in observations[0]:
                assert all(
                    first <= last
                    for first, last in zip(
                        observations[0]["energyPoints"], observations[-1]["energyPoints"], strict=True
                    )
                ), observations
                assert observations[0]["weights"] != observations[-1]["weights"], observations
            elif "completed" in observations[0]:
                assert 0 < observations[0]["compiled"] <= observations[-1]["compiled"], observations
                assert observations[0]["completed"] < observations[-1]["completed"], observations
                assert observations[0]["count"] < observations[-1]["count"], observations
                assert any(o["completed"] != o["count"] for o in observations), observations
            else:
                counts = [o["count"] for o in observations]
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
                    page.evaluate("MQSF_DECK.finishAnimation()")
                    assert not assert_state(expected)["animating"], "Finishing a replay must preserve its build"
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

        motion_layout_samples = 0
        for index, slide in enumerate(slides):
            for step, playback in slide["playbacks"].items():
                page.evaluate("([i,s]) => MQSF_DECK.go(i,s,true)", [index, int(step)])
                page.clock.run_for(playback.get("transition", 0) + 32)
                previous_fraction = 0
                for fraction in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95):
                    page.clock.fast_forward(round(playback["duration"] * (fraction - previous_fraction)))
                    for issue in page.evaluate(LAYOUT_CHECK):
                        geometry.setdefault((index, issue), int(step))
                    inspect_evidence()
                    motion_layout_samples += 1
                    previous_fraction = fraction

        page.evaluate("dispatchEvent(new Event('beforeprint'))")
        page.emulate_media(media="print")
        assert page.locator("#print-deck .print-slide").count() == len(slides)
        assert page.locator("#deck").is_hidden()
        for index in range(len(slides)):
            printed = page.locator(".print-slide").nth(index)
            assert printed.locator(".slide-footer").inner_text().endswith(f"{index + 1} / {len(slides)}")
            assert printed.locator("h1, h2").count() > 0
            print_layout = LAYOUT_CHECK.replace(
                "document.getElementById('slide')",
                f"document.querySelector('.print-slide:nth-child({index + 1}) article')",
            )
            for issue in page.evaluate(print_layout):
                geometry.setdefault((index, f"print: {issue}"), slides[index]["builds"])
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
        stability = page.evaluate(STABILITY_CHECK)
        assert not stability["failures"], stability["failures"]
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
            f"exact result progress, {len(slides)} print pages, {motion_layout_samples} animation layout samples."
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
