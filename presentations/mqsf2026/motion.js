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
  const entrances = new Set();
  let cancel = () => {};
  function finish() {
    cancel();
    entrances.forEach((animation) => animation.cancel());
    entrances.clear();
  }
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
    entrances.forEach((animation) => animation.cancel());
    entrances.clear();
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!duration || reduced) {
      cancel();
      root.innerHTML = html;
      return;
    }
    const old = new Map(
      [...root.querySelectorAll("[data-morph]")].map((n) => [
        n.dataset.morph,
        { node: n, rect: n.getBoundingClientRect() },
      ]),
    );
    cancel(false);
    root.innerHTML = html;
    const tweens = [],
      animations = [];
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
        else {
          const opacity = Number(getComputedStyle(node).opacity);
          // Hidden future builds must stay hidden during their entry animation.
          if (opacity > 0)
            animations.push(
              node.animate([{ opacity: 0 }, { opacity }], {
                duration: 650,
                fill: "backwards",
              }),
            );
        }
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
    let begin, frame;
    const finish = (settle = true) => {
      cancelAnimationFrame(frame);
      if (settle)
        tweens.forEach(({ node, name, target }) =>
          node.setAttribute(name, target),
        );
      animations.forEach((a) => a.cancel());
      cancel = () => {};
    };
    cancel = finish;
    function tick(now) {
      begin ??= now;
      const fraction = Math.max(0, Math.min(1, (now - begin) / duration)),
        ease = 1 - Math.pow(1 - fraction, 4);
      tweens.forEach(({ node, name, lerp }) =>
        node.setAttribute(name, lerp(ease)),
      );
      if (fraction < 1) frame = requestAnimationFrame(tick);
      else finish();
    }
    frame = requestAnimationFrame(tick);
  }
  function update(root, html) {
    const template = document.createElement("template");
    template.innerHTML = html;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    function enter(node) {
      if (reduced || node.nodeType !== Node.ELEMENT_NODE) return;
      const keyed = node.matches("[data-morph]")
        ? [node]
        : node.querySelectorAll("[data-morph]");
      for (const child of keyed) {
        if (!(child instanceof SVGElement)) continue;
        const opacity = Number(getComputedStyle(child).opacity);
        if (!opacity) continue;
        const animation = child.animate([{ opacity: 0 }, { opacity }], {
          duration: 200,
          easing: "ease-out",
          fill: "backwards",
        });
        entrances.add(animation);
        animation.onfinish = animation.oncancel = () =>
          entrances.delete(animation);
      }
    }
    function sync(parent, content) {
      for (let i = 0; i < content.childNodes.length; i++) {
        const target = content.childNodes[i],
          current = parent.childNodes[i];
        if (!current) {
          const added = target.cloneNode(true);
          parent.append(added);
          enter(added);
        } else if (
          current.nodeType !== target.nodeType ||
          current.nodeName !== target.nodeName ||
          current.namespaceURI !== target.namespaceURI ||
          current.dataset?.morph !== target.dataset?.morph
        ) {
          const added = target.cloneNode(true);
          current.replaceWith(added);
          enter(added);
        } else if (target.nodeType === Node.ELEMENT_NODE) {
          for (const { name } of [...current.attributes])
            if (!target.hasAttribute(name)) current.removeAttribute(name);
          for (const { name, value } of target.attributes)
            if (current.getAttribute(name) !== value) {
              if (name === "opacity")
                current
                  .getAnimations()
                  .filter((animation) => entrances.has(animation))
                  .forEach((animation) => animation.cancel());
              current.setAttribute(name, value);
            }
          sync(current, target);
        } else if (current.nodeValue !== target.nodeValue)
          current.nodeValue = target.nodeValue;
      }
      while (parent.childNodes.length > content.childNodes.length)
        parent.lastChild.remove();
    }
    // Keep stable SVG nodes alive so their motion does not restart every frame.
    sync(root, template.content);
  }
  return { replace, update, finish, interpolate };
})();
