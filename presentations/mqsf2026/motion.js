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
window.MQSF_MOTION = (() => {
  const number = /[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:e[-+]?\d+)?/gi;
  let cancel = () => {};
  function interpolate(from, to) {
    const a = from.match(number)?.map(Number),
      b = to.match(number)?.map(Number);
    if (
      !a ||
      !b ||
      a.length !== b.length ||
      from.replace(number, "#") !== to.replace(number, "#")
    )
      return null;
    return (t) => {
      let i = 0;
      return to.replace(number, () => String(a[i] + (b[i] - a[i++]) * t));
    };
  }
  function replace(root, html, duration = 950) {
    cancel();
    const old = new Map(
      [...root.querySelectorAll("[data-morph]")].map((n) => [
        n.dataset.morph,
        { node: n, rect: n.getBoundingClientRect() },
      ]),
    );
    root.innerHTML = html;
    const tweens = [],
      animations = [];
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const svgAttributes = [
      "x",
      "y",
      "x1",
      "y1",
      "x2",
      "y2",
      "cx",
      "cy",
      "r",
      "rx",
      "width",
      "height",
      "opacity",
      "fill-opacity",
      "stroke-width",
      "transform",
      "d",
      "points",
      "viewBox",
    ];
    if (!duration || reduced) return;
    function pair(before, after) {
      if (before.tagName !== after.tagName) return;
      for (const name of svgAttributes) {
        const target = after.getAttribute(name),
          source = before.getAttribute(name);
        if (target === source || target == null || source == null) continue;
        const lerp = interpolate(source, target);
        if (lerp) {
          after.setAttribute(name, source);
          tweens.push({ node: after, name, target, lerp });
        }
      }
      const a = [...before.children],
        b = [...after.children];
      b.forEach((node, i) => {
        if (!node.dataset.morph && a[i]) pair(a[i], node);
      });
    }
    for (const node of root.querySelectorAll("[data-morph]")) {
      const previous = old.get(node.dataset.morph);
      if (node instanceof SVGElement) {
        if (previous) pair(previous.node, node);
        else
          animations.push(
            node.animate([{ opacity: 0 }, { opacity: 1 }], {
              duration: 650,
              fill: "backwards",
            }),
          );
      } else if (previous) {
        const current = node.getBoundingClientRect(),
          scale =
            Number(
              getComputedStyle(document.documentElement).getPropertyValue(
                "--scale",
              ),
            ) || 1;
        const dx = (previous.rect.x - current.x) / scale,
          dy = (previous.rect.y - current.y) / scale;
        if (current.width && current.height)
          animations.push(
            node.animate(
              [
                {
                  transform: `translate(${dx}px,${dy}px) scale(${previous.rect.width / current.width},${previous.rect.height / current.height})`,
                  transformOrigin: "0 0",
                },
                { transform: "none", transformOrigin: "0 0" },
              ],
              { duration, easing: "cubic-bezier(.22,.75,.18,1)" },
            ),
          );
        if (node.textContent !== previous.node.textContent)
          animations.push(
            node.animate([{ opacity: 0.15 }, { opacity: 1 }], {
              duration: 550,
              easing: "ease-out",
            }),
          );
      }
    }
    // New content enters gently; shared SVG objects follow their actual geometry.
    if (!old.size)
      animations.push(
        root.animate(
          [
            { opacity: 0, transform: "translateY(14px)" },
            { opacity: 1, transform: "none" },
          ],
          { duration: 450 },
        ),
      );
    const begin = performance.now();
    let frame;
    const finish = () => {
      cancelAnimationFrame(frame);
      tweens.forEach(({ node, name, target }) =>
        node.setAttribute(name, target),
      );
      animations.forEach((a) => a.cancel());
      cancel = () => {};
    };
    cancel = finish;
    function tick(now) {
      const fraction = Math.min(1, (now - begin) / duration),
        ease = 1 - Math.pow(1 - fraction, 4);
      tweens.forEach(({ node, name, lerp }) =>
        node.setAttribute(name, lerp(ease)),
      );
      if (fraction < 1) frame = requestAnimationFrame(tick);
      else finish();
    }
    frame = requestAnimationFrame(tick);
  }
  return { replace, finish: () => cancel(), interpolate };
})();
