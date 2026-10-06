# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check the packaged presentation in a browser with networking disabled."""

from __future__ import annotations

import argparse
import gzip
from base64 import b64decode
from importlib import import_module
from pathlib import Path
from typing import Any

TITLE = "System Software for Quantum Computing: From the Metal to the User"
DEFAULT_HTML = Path(__file__).resolve().parents[2] / "build/mqsf2026/index.html"


def check(html: Path, executable: str | None, screenshot: Path | None) -> None:
    """Verify navigation, capture selection, replay, and complete source downloads."""
    with import_module("playwright.sync_api").sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=executable, args=["--no-sandbox"])
        context = browser.new_context(offline=True, viewport={"width": 1440, "height": 900}, reduced_motion="reduce")
        network: list[str] = []
        errors: list[str] = []

        def route_request(route: Any) -> None:  # ruff: ignore[any-type] - Playwright is an optional runtime import.
            if route.request.url.startswith(("http:", "https:")):
                network.append(route.request.url)
                route.abort()
            else:
                route.continue_()

        context.route("**/*", route_request)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(html.resolve().as_uri(), wait_until="load")
        assert page.title() == TITLE
        assert page.locator("#chapters button").count() == 7
        chapters = ["opening", "compiler", "architecture", "execution", "application", "benchmarks", "closing"]
        for index, chapter in enumerate(chapters):
            page.locator("#chapters button").nth(index).click()
            assert page.locator(f"#{chapter}").is_visible()
            assert page.locator(".chapter:visible").count() == 1

        cases = page.evaluate("""() => window.MQSF_DATA.scenarios.flatMap((scenario, si) =>
            scenario.variants.filter(variant => !variant.unavailable_reason).map(variant => ({
                scenario: si, variant: scenario.variants.indexOf(variant), name: `${scenario.id}/${variant.id}`,
                exports: variant.exports.filter(output => typeof output.code === 'string' && !output.unavailable_reason)
                    .map(output => ({id: output.id, sha256: output.sha256, execution: variant.executions?.[output.id] ||
                        (output.sha256 === variant.execution?.payload_sha256 ? variant.execution : null)})),
                default_payload: variant.execution?.payload_sha256 || null,
                counts: variant.execution?.counts || null,
                shots: variant.execution?.shots?.length || 0
            })))""")
        assert cases, "No selectable compiler captures"
        execution_cases = 0
        for case in cases:
            if case["shots"]:
                assert case["default_payload"], f"Successful default capture has no payload hash: {case['name']}"
                assert any(output.get("sha256") == case["default_payload"] for output in case["exports"]), (
                    f"Successful default capture has no matching selectable export: {case['name']}"
                )
            default_exercised = False
            page.locator("#chapters button").nth(1).click()
            page.locator("#scenario").select_option(str(case["scenario"]))
            page.locator("#variant").select_option(str(case["variant"]))
            stage_count = page.locator("#pipeline button").count()
            assert stage_count > 0
            page.locator("#pipeline button").nth(stage_count - 1).click()
            assert page.locator("#code").inner_text().strip()
            for output in case["exports"]:
                page.locator("#output-format").select_option(output["id"])
                assert page.locator("#code").inner_text().strip()
                page.locator("#chapters button").nth(3).click()
                captured = output["execution"]
                if captured and not captured.get("unavailable_reason"):
                    assert page.locator("#execution-format").inner_text() == captured["format"]
                    assert page.locator("#execution-payload-hash").inner_text() == captured["payload_sha256"]
                    duration = page.evaluate(
                        "duration => duration.toLocaleString(undefined, {maximumFractionDigits: 2})",
                        captured["duration_ms"],
                    )
                    assert f"{duration} ms" in page.locator("#execution-timing").inner_text()
                    assert page.locator("#execution-device").inner_text()
                    if captured["payload_sha256"] == case["default_payload"]:
                        default_exercised = True
                else:
                    assert page.locator("#replay-toggle").is_disabled()
                    assert page.locator("#execution-empty").is_visible()
                page.locator("#chapters button").nth(1).click()
            if case["shots"]:
                assert default_exercised, f"Default capture was not exercised through its export: {case['name']}"
            page.locator("#output-format").select_option("stage")
            page.locator("#chapters button").nth(2).click()
            expected_sites = page.evaluate("window.MQSF_DATA.device.sites.length")
            assert page.locator(".physical-site").count() == expected_sites
            if case["shots"]:
                execution_cases += 1
                page.locator("#chapters button").nth(3).click()
                page.locator("#replay-toggle").click()
                assert page.locator("#replay-toggle").inner_text() == "Pause"
                page.locator("#replay-toggle").click()
                page.locator("#replay-position").evaluate(
                    "node => { node.value = '1000'; node.dispatchEvent(new Event('input')); }"
                )
                expected = [case["counts"][key] for key in sorted(case["counts"])]
                assert [int(text) for text in page.locator(".hist-count").all_text_contents()] == expected
                assert page.locator(".sequence-event:not(.visible)").count() == 0
                assert page.locator("#shot-progress").inner_text().endswith(f"/ {case['shots']:,} shots")
                page.locator("#replay-reset").click()
                assert all(int(text) == 0 for text in page.locator(".hist-count").all_text_contents())
        assert execution_cases, "No ordered execution capture was exercised"

        page.locator("#chapters button").nth(1).click()
        page.locator("#scenario").select_option(str(cases[0]["scenario"]))
        page.locator("#variant").select_option(str(cases[0]["variant"]))
        page.locator("#pipeline button").first.click()
        artifact = page.evaluate("""() => MQSF_DATA.scenarios[Number(document.getElementById('scenario').value)]
            .variants[Number(document.getElementById('variant').value)].stages[0]""")
        expected_source = (
            gzip.decompress(b64decode(artifact["full_code_gzip"])).decode()
            if artifact.get("full_code_gzip")
            else artifact["code"]
        )
        with page.expect_download() as download_event:
            page.locator("#download-code").click()
        download_path = download_event.value.path()
        assert download_path is not None
        assert Path(download_path).read_text(encoding="utf-8") == expected_source
        if artifact.get("full_code_gzip"):
            page.locator("#full-code").click()
            page.wait_for_function("document.getElementById('full-code').textContent === 'Show excerpt'")
            assert page.locator("#code").inner_text() == expected_source
            page.locator("#full-code").click()
            assert page.locator("#full-code").inner_text() == "Full source"

        page.locator("#main").focus()
        page.keyboard.press("End")
        assert page.locator("#closing").is_visible()
        page.keyboard.press("Home")
        assert page.locator("#opening").is_visible()
        page.reload(wait_until="load")
        assert page.locator("#opening").is_visible()
        if screenshot:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot), full_page=True)
        assert not errors, f"Browser errors: {errors}"
        assert not network, f"Presentation requested network resources: {network}"
        context.close()
        browser.close()
        print(f"Offline browser checks passed: {len(cases)} compiler variants, {execution_cases} execution captures.")


def main() -> None:
    """Run the standalone browser check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--browser", help="Chromium executable; omit for Playwright's bundled browser")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    check(args.html, args.browser, args.screenshot)


if __name__ == "__main__":
    main()
