# Final handoff checks

Core baseline `1d384a5d23aa1f02b26fd5684dab586ef26167fe`, uncommitted
`feat/native-equivalence` implementation, 2026-10-25. The pinned differential
record identifies QCEC and its separate compatible Core dependency.

| Check                                | Expected                                                           | Observed                                                                                                               | Evidence                                                               |
| ------------------------------------ | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Affected native suites               | All existing and new tests pass                                    | 2133/2133 across eight binaries                                                                                        | [Native record](native-tests.md)                                       |
| Broadened Python suites              | Existing MLIR/DD and native API contracts pass                     | 477/477                                                                                                                | Command below; local log `/tmp/native-equivalence-final-python.log`    |
| Generated stubs and focused bindings | Rebuild succeeds and typed binding tests pass                      | Success; 34/34                                                                                                         | [Binding record](README.md)                                            |
| Independent dense oracle             | Phase, normalized distances, effects, ancillas and mutations agree | Six tests pass, within the 300-test Utils suite                                                                        | [Native record](native-tests.md)                                       |
| Pinned differential                  | Declared overlap agrees, contract differences remain explicit      | All 11 expected native/QCEC pairs                                                                                      | [Differential](../qcec-differential/README.md)                         |
| Executable documentation             | Build, notebooks, references and internal links succeed            | Passed                                                                                                                 | Commands below                                                         |
| Rendered API page                    | Tables/math readable; no overflow or page errors                   | Inspected screenshot; width and scroll width both 1905; no MathJax or console errors; 40/40 requests return 200        | Chrome DevTools, local preview                                         |
| Repository lint                      | All hooks pass without modifying files                             | Passed; no files modified                                                                                              | `uvx nox -s lint`; local log `/tmp/native-equivalence-final-lint7.log` |
| Full-file C++ lint                   | All changed C++ files checked without findings                     | Passed; 28 implementation/test TUs plus the test helper header, zero diagnostics; 42 changed C++ file hashes unchanged | [Lint record](README.md) and `cpp-lint-files.txt`                      |

Broadened Python command, after the final `uvx nox -s stubs` rebuild:

```sh
uv pip install --python .nox/stubs/bin/python --group test-base
MQT_CORE_QDMI_CONFIG_FILE="$PWD/build/release/src/qdmi/mqt-core-qdmi-sc-device.qdmi.json" \
  .nox/stubs/bin/python -m pytest -q -n4 \
  test/python/test_qco_dd.py test/python/dd test/python/test_mlir.py \
  test/python/test_mlir_loops.py test/python/test_mlir_parameter_binding.py \
  test/python/test_verification.py
```

An earlier broader run had six optional-SC-provider setup failures; supplying
the local provider resolved all of them. Stub regeneration synchronizes its venv
and removes extra test dependencies, so the test group was restored before the
final run. The first final invocation failed before collection because
pytest-xdist was absent; no test result was inferred from that invocation.

Documentation commands:

```sh
uvx nox --non-interactive -s docs
.nox/docs/bin/sphinx-build -n -T -W -b html \
  -d docs/_build/doctrees/html docs docs/_build/html
.nox/docs/bin/python scripts/check_docs_links.py docs/_build/html
```

The first command ran all executable documentation successfully. The incremental
second command included the final cancellation-limit clarification; its render
and internal links were checked afterward. Logs are
`/tmp/native-equivalence-final-docs.log` and
`/tmp/native-equivalence-final-docs-incremental.log`. The preview used
`python3 -m http.server 8765 --bind 127.0.0.1 --directory docs/_build/html`. An
initial preview requested a missing favicon; the final fresh navigation had no
console messages and all 40 requests succeeded. No hosted CI, full-repository
test run, or representative performance benchmark is claimed.
