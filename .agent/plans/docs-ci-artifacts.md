# Documentation builds and artifact publication

Status: executable documentation and live PR artifact publication passed.
Automatic publication and the required-check migration await merge to main.
Evidence and rollout status are tracked in PR #2922.

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
- [x] Measure cold and cached builds and verify generated output.
- [x] Review the final change and submit a signed PR.
- [x] Verify hosted publication after RtD activation and token provisioning.

## Measurement baseline

RtD build 35620474 ran `nox -s docs` in 609 seconds; the 18 reported notebook
execution times sum to 73.11 seconds. Dispatch waited 1085 seconds. The build
summary's end time differs from command records, so do not combine them into a
precise total. Separate queue time, active build time, and publication time.
Comparisons across RtD and GitHub runners include the hardware change.

Local Spark validation (AArch64, four compilation jobs): the first build with an
empty compiler cache took 271.36 seconds, including 183.31 seconds for the
package and 87.30 seconds for rendering and checks. All 18 notebooks executed.
Python downloads were already cached; this is not a cold network measurement.
The six focused tests and repository lint passed. Four targeted mutations
confirmed commit, stale-PR, deleted-fork, and symlink regression coverage.
Chrome inspection confirmed rendered notebook output and loaded styles/scripts;
its only failed request was the browser's implicit favicon request.

The first hosted build (run 41695361664) passed all 18 notebooks with no
compiler cache hits: 418.25 seconds for docs (312.13 package, 103.86
rendering/checks), 597 seconds for the job. Checkout alone took 112 seconds. A
blob-filtered clone preserves the release tag and commit distance while omitting
historical file contents; the local clone used 15.18 MiB of Git packs. Run
41698109059 then passed in 237 seconds (161.08 seconds for docs), with all 18
notebooks executed and 145/145 compiler-cache hits. Checkout took 24 seconds.

External linkcheck reported two missing anchors on existing QIR 2.1 URLs amid
GitHub rate limits. Both exact headings exist in the upstream source. Local HTML
links and assets passed. The pinned RtD client produced a valid 17.58 MiB upload
archive containing the site and offline ZIP.

On November 10, the project owner confirmed direct upload is enabled for all MQT
projects. After rebasing onto main, run 42769682355 passed all 18 notebooks in
510 seconds (425.28 seconds for docs, 81/150 compiler-cache hits). The owner
provided a private token file; the repository secret and enable flag are set.
The same pinned CLI uploaded that Actions artifact as PR preview 2922 in eight
seconds. RtD build 35776729 finished successfully with the exact commit SHA, and
GitHub received a successful RtD status. The local validation helper checked the
current PR/run identity before upload; the automatic workflow itself still
requires merge to the default branch.

Chrome confirmed the rendered site and its documentation assets. The published
ZIP matches the Actions artifact byte-for-byte. The RtD flyout currently links
PR downloads to the production hostname (404); the same path on the preview
hostname works. This is a hosting-side link issue, not a missing artifact. The
existing `latest` and `stable` targets are unchanged. Keep the existing
required-check ruleset until the workflow is on main and automatic branch
publication has been verified.
