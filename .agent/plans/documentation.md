# Executable documentation

Status: implementation replayed on upstream main `0c3fac2cb` and locally
validated.

## Goal and decisions

Resolve all findings in `.agent/audits/documentation.md` and submit them for
review. Keep MyST notebooks, bundled DDSIM fixtures, Sphinx, and native Doxygen.
Correct template-owned guidance upstream and render the affected Core pages.

- Force execution, check command failures, and isolate the QDMI registry.
- Check published local navigation. Select Doxygen's static menus to avoid its
  malformed dynamic member-index links; do not patch generated HTML.
- Keep exhaustive interoperability contracts in a linked reference. Tutorials
  display results and assert semantic invariants.
- Keep remote-provider deployment examples illustrative; no hardware execution
  is needed to validate these documentation changes.

## Validation and publication

- `uvx nox --non-interactive -s docs`: passed, including the generated HTML link
  checker. Nine notebooks contain 55 code cells; none produced error or stderr
  outputs.
- `uvx nox --non-interactive -s docs -- -b linkcheck`: passed with the
  repository's existing external-link exclusions.
- `uvx nox -s lint`: passed. Generated-navigation regression tests: 2 passed.
- `uvx nox -s cpp-lint -- origin/main`: completed, but its configured
  public-header exclusion selected no files. The earlier separate full-header
  cpp-linter run checked `Client.hpp`: five existing naming/conversion warnings,
  reproduced against the original header. No new diagnostic or runtime change
  was introduced.
- The full HTML session passes with Read the Docs' Doxygen 1.9.8 after adding
  the public include root and removing unsupported custom-header markers.
  Earlier Doxygen 1.17 validation preserved all 157 compound and 890 member
  entries in the reduced configuration's tag inventory.
- Inspected the regenerated DD and QAOA figures, rendered image alternatives,
  and C++ landing page in the generated agent index. The final browser preview
  could not connect to localhost; no full responsive-browser claim is made.
- All 13 upstream template rendering tests and full lint passed. Template
  changes were merged in
  [Templates #438](https://github.com/munich-quantum-toolkit/templates/pull/438);
  the workflow pins its main commit `38a2369c8303887135af5e98d9e7003ed267b613`.
- Commits are signed and verified. Hosted CI and human review remain pending.

The companion audit records the dispositions of all twelve finding groups.
Provider deployment and remote hardware were outside local validation.
