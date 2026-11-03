# Documentation builds and artifact publication

Status: implementation and local validation complete; hosted measurements and
RtD activation remain.

## Goal and scope

Build the package and complete executable documentation in GitHub Actions.
Publish the resulting HTML and offline ZIP to Read the Docs without rebuilding
them. Keep notebook execution, warning failures, native references, and local
link and asset checks in the existing `nox -s docs` entry point.

## Decisions

- A standalone Documentation workflow runs independently of the full CI matrix.
  Its build check replaces the RtD status as the required documentation check
  during activation. Do not add cross-workflow polling to the CI aggregate.
- Pull requests build their head commit; branch and tag builds use their event
  commit. A manifest and GitHub run metadata identify the published source.
- A trusted `workflow_run` publisher handles successful artifacts as data. It
  never executes PR code or supplies publication credentials to the builder. It
  rejects mismatched commits and skips superseded or closed PRs and moved refs.
  Merge groups are validated but not published.
- Use the official RtD direct-upload client at an exact revision. The current
  upload action installs its client from main, so pinning only that action would
  not pin the code receiving the token.
- RtD direct upload requires support activation and an admin token. Preserve the
  existing project mode and old-version source build configuration until
  activation. Uploading an existing version switches it to artifact updates.

## Work remaining

- [x] Implement and test the builder and trusted artifact publisher.
- [ ] Measure cold and cached builds and verify generated output.
- [ ] Review the final change and submit a signed PR.
- [ ] Verify hosted publication after RtD activation and token provisioning.

## Measurement baseline

RtD build 35620474 ran `nox -s docs` in 609 seconds; the 18 reported notebook
execution times sum to 73.11 seconds. Dispatch waited 1085 seconds. The build
summary's end time differs from command records, so do not combine them into a
precise total. Separate queue time, active build time, and publication time.
Comparisons across RtD and GitHub runners include the hardware change.

Local Spark validation (AArch64, four compilation jobs): the first build with
an empty compiler cache took 271.36 seconds, including 183.31 seconds for the
package and 87.30 seconds for rendering and checks. All 18 notebooks executed.
Python downloads were already cached; this is not a cold network measurement.
The six focused tests and repository lint passed. Four targeted mutations
confirmed commit, stale-PR, deleted-fork, and symlink regression coverage.
Chrome inspection confirmed rendered notebook output and loaded styles/scripts;
its only failed request was the browser's implicit favicon request.
