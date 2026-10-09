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


def check(html: Path, executable: str | None, screenshot: Path | None, screenshots_dir: Path | None = None) -> None:
    """Exercise every build, keyboard navigation, replay completion, and visible assets.

    Layout findings are collected across the whole deck and reported together,
    after the functional checks, so one overflow does not hide other failures.
    """
    with import_module("playwright.sync_api").sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=executable, args=["--no-sandbox"])
        context = browser.new_context(offline=True, viewport={"width": 1920, "height": 1080}, reduced_motion="reduce")
        network: list[str] = []
        errors: list[str] = []
        geometry: dict[tuple[int, str], int] = {}

        def route_request(route: Any) -> None:  # ruff: ignore[any-type] - Playwright is an optional runtime import.
            if route.request.url.startswith(("http:", "https:")):
                network.append(route.request.url)
                route.abort()
            else:
                route.continue_()

        context.route("**/*", route_request)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        # Slow CI must not miss the short 1x replay before checking its next-click behavior.
        page.clock.install(time="2026-10-14T07:00:00Z")
        page.goto(html.resolve().as_uri(), wait_until="load")
        page.evaluate("document.fonts.ready")
        page.clock.pause_at("2026-10-14T08:00:00Z")
        assert page.title() == TITLE
        slides = page.evaluate("window.MQSF_DECK.slides")
        assert len(slides) == 28, f"Expected the 28-slide keynote, found {len(slides)} slides"
        states = [(index, step) for index, slide in enumerate(slides) for step in range(slide["builds"] + 1)]
        execution = page.evaluate("""() => {
            const run = MQSF_DATA.scenarios.find(s => s.id === 'qpe').variants[0].execution;
            const end = Math.max(run.completed_ms, ...run.events.map(e => e.time_ms + e.duration_ms));
            return {shots: run.num_shots, halfway: run.shot_events.filter(s => s.time_ms <= end / 2).length};
        }""")
        shots = execution["shots"]

        def assert_state(expected: tuple[int, int]) -> dict[str, Any]:
            current = page.evaluate("window.MQSF_DECK.getState()")
            assert (current["slide"], current["step"]) == expected, (expected, current)
            assert page.locator("#slide").get_attribute("data-slide") == str(expected[0] + 1)
            assert page.locator("#slide").get_attribute("data-step") == str(expected[1])
            return current

        def inspect_slide(index: int, step: int) -> None:
            for issue in page.evaluate(LAYOUT_CHECK):
                geometry.setdefault((index, issue), step)
            assert page.locator("#slide-number").inner_text().startswith(f"{index + 1:02d} / {len(slides)}")
            assert page.locator("#section-label").inner_text().strip()
            assert page.locator("#mqsc-logo").is_visible()
            if step != slides[index]["builds"]:
                return
            page.wait_for_function(
                "[...document.querySelectorAll('#deck img')].every(i => i.complete && i.naturalWidth > 0)"
            )
            assets = page.evaluate("""() => [...document.querySelectorAll('#deck img, #deck svg image')].map(i => ({
                source: i.currentSrc || i.getAttribute('href') || i.getAttribute('xlink:href')
            }))""")
            assert assets, f"Slide {index + 1} has no branding image"
            assert all(asset["source"].startswith("data:image/svg+xml") for asset in assets), (
                f"Slide {index + 1} contains a non-vector or external image"
            )
            assert page.locator("#deck canvas, #deck video").count() == 0
            if screenshots_dir:
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshots_dir / f"slide-{index + 1:02d}.png"))

        page.keyboard.press("Home")
        completed_animations = 0
        for position, expected in enumerate(states):
            current = assert_state(expected)
            if current["animating"]:
                if "slow down" in slides[expected[0]]["title"]:
                    page.clock.fast_forward(9000)
                    progress = page.locator(".histogram").evaluate("""node => ({
                        shots: Number(node.dataset.shots),
                        bars: [...node.querySelectorAll('[data-count]')]
                            .reduce((n, bar) => n + Number(bar.dataset.count), 0)
                    })""")
                    assert 0 < progress["shots"] < shots
                    assert progress["shots"] == execution["halfway"], "Replay must follow the actual shot timestamps"
                    assert progress["shots"] == progress["bars"]
                    assert "device_job_wait" in page.locator(".code-line.hot").inner_text()
                    for issue in page.evaluate(LAYOUT_CHECK):
                        geometry.setdefault((expected[0], issue), expected[1])
                    if screenshots_dir:
                        page.screenshot(path=str(screenshots_dir / "runtime-progress.png"))
                page.keyboard.press("PageDown")
                assert assert_state(expected)["animating"] is False, "First click must finish, not advance, a replay"
                assert page.locator(".histogram").get_attribute("data-shots") == str(shots)
                completed_animations += 1
            inspect_slide(*expected)
            if position + 1 < len(states):
                page.keyboard.press(("PageDown", "Space", "ArrowRight")[position % 3])
        assert completed_animations == 2, f"Expected both runtime replays, observed {completed_animations}"
        page.keyboard.press("PageDown")
        assert_state(states[-1])
        for expected in reversed(states[:-1]):
            page.keyboard.press("PageUp")
            assert assert_state(expected)["animating"] is False
        page.keyboard.press("PageUp")
        assert_state(states[0])

        page.keyboard.press("End")
        assert_state(states[-1])
        page.keyboard.press("Home")
        assert_state(states[0])
        page.keyboard.press("2")
        page.keyboard.press("8")
        assert page.locator("#jump-indicator").is_visible()
        page.keyboard.press("Enter")
        assert_state((27, 0))
        assert page.locator("#jump-indicator").is_hidden()
        page.reload(wait_until="load")
        assert_state((27, 0))

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
        assert page.locator("#overview").is_visible()
        page.keyboard.press("ArrowLeft")
        page.keyboard.press("Escape")
        assert page.locator("#overview").is_hidden()
        assert_state((6, 0))
        page.keyboard.press("b")
        assert page.locator("#blackout").is_visible()
        page.keyboard.press("b")
        assert page.locator("#blackout").is_hidden()
        page.keyboard.press("p")
        assert page.locator("#speaker-notes").is_visible()
        page.keyboard.press("p")
        assert page.locator("#speaker-notes").is_hidden()

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
        for key in ["PageDown"] * 16 + ["PageUp"] * 7:
            page.keyboard.press(key)
        expected = states[9]
        page.wait_for_function(
            "expected => {const slide = document.getElementById('slide');"
            "return Number(slide.dataset.slide) === expected[0] + 1 && Number(slide.dataset.step) === expected[1];}",
            arg=expected,
        )
        page.wait_for_function("document.getAnimations().every(animation => animation.playState !== 'running')")
        assert_state(expected)
        assert not errors, f"Browser errors: {errors}"
        assert not network, f"Presentation requested network resources: {network}"
        context.close()
        browser.close()
        print(f"Offline browser checks passed: {len(slides)} slides, {len(states)} builds, both timed replays.")
        for (index, issue), step in geometry.items():
            print(f"Layout: slide {index + 1}, build {step}: {issue}")
        assert not geometry, f"Found {len(geometry)} FullHD layout issues (listed above)"


def main() -> None:
    """Run the standalone browser check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--browser", default=os.environ.get("MQSF_BROWSER") or os.environ.get("CHROME_BIN"))
    parser.add_argument("--screenshot", type=Path, help="Save the opening slide")
    parser.add_argument("--screenshots-dir", type=Path, help="Save the final build of every slide at FullHD")
    args = parser.parse_args()
    check(args.html, args.browser, args.screenshot, args.screenshots_dir)


if __name__ == "__main__":
    main()
