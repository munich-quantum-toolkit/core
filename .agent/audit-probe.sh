#!/usr/bin/env sh
# Copyright (c) 2026 Chair for Design Automation, TUM
# Copyright (c) 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Measure how much a single test assertion actually protects.
#
# See .agent/AUDITS.md for the method this script implements. It runs the two
# cheap evidence tiers of a SpecAudit and prints a block to paste into the
# audit ledger.
#
#   t1  coverage delta   drop a test, see whether coverage of the source moves
#   t2  fault injection  break the code, see which tests notice
#
# The script refuses to run on a dirty working tree and restores every file it
# edits.

set -eu

script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH= cd -- "${script_directory}/.." && pwd)

edited_file=""

usage() {
  cat <<'EOF'
Usage:
  .agent/audit-probe.sh t1 --lang python --source SRC --tests TESTS --drop NODE
  .agent/audit-probe.sh t1 --lang cpp --source SRC --target TGT [--ctest RE]
  .agent/audit-probe.sh t2 --lang python --tests TESTS --inject FILE:LINE
                           --with TEXT
  .agent/audit-probe.sh t2 --lang cpp --target TGT --inject FILE:LINE
                           --with TEXT [--ctest RE]

Tiers:
  t1  Coverage delta. Runs the suite, then runs it again without the dropped
      test, and reports the coverage of SRC in both runs. No movement is
      evidence of redundancy. It is never proof on its own.
  t2  Fault injection. Replaces one line of FILE with TEXT, runs the suite,
      and reports which tests failed. This is the tier that settles verdicts.

Options:
  --lang     python or cpp.
  --source   Repository-relative path of the code under audit.
  --tests    Repository-relative test path passed to pytest.
  --target   CMake target to build for the C++ suite.
  --ctest    Regular expression passed to ctest -R.
  --drop     A pytest node id to deselect, for example
             test/python/x/test_y.py::test_z
  --inject   FILE:LINE to overwrite, repository-relative.
  --with     Replacement text for that line. Use an empty string to delete it.
  --keep     Leave the injected change in place instead of restoring it.

Every command uses the build and test entry points AGENTS.md documents. A C++
probe needs a configured coverage preset; see AGENTS.md.
EOF
}

fail() {
  echo "audit-probe: $1" >&2
  exit 2
}

restore_edit() {
  if [ -n "${edited_file}" ] && [ "${keep_edit}" != "yes" ]; then
    git -C "${repository_root}" checkout -- "${edited_file}"
    edited_file=""
  fi
}

require_clean_tree() {
  if [ -n "$(git -C "${repository_root}" status --porcelain)" ]; then
    fail "working tree is not clean; commit or stash before probing"
  fi
}

# Replace one line of a file, in place.
overwrite_line() {
  file="$1"
  line="$2"
  text="$3"
  [ -f "${repository_root}/${file}" ] || fail "no such file: ${file}"
  total=$(wc -l <"${repository_root}/${file}")
  [ "${line}" -ge 1 ] 2>/dev/null || fail "line must be a positive number"
  [ "${line}" -le "${total}" ] || fail "${file} has only ${total} lines"
  awk -v target="${line}" -v replacement="${text}" \
    'NR == target { print replacement; next } { print }' \
    "${repository_root}/${file}" >"${repository_root}/${file}.probe-tmp"
  mv "${repository_root}/${file}.probe-tmp" "${repository_root}/${file}"
  edited_file="${file}"
}

python_coverage() {
  # $1 source path, $2 tests path, $3 optional pytest node id to deselect
  cd "${repository_root}"
  if [ -n "$3" ]; then
    uv run --no-sync pytest "$2" --deselect "$3" \
      --cov-config=pyproject.toml --cov -q >/dev/null 2>&1 || true
  else
    uv run --no-sync pytest "$2" \
      --cov-config=pyproject.toml --cov -q >/dev/null 2>&1 || true
  fi
  uv run --no-sync coverage report \
    --include="$1/*" --precision=2 2>/dev/null | tail -n 1
}

python_suite() {
  # $1 tests path. Prints failing node ids, one per line.
  cd "${repository_root}"
  uv run --no-sync pytest "$1" -q -p no:randomly \
    --no-header -rf 2>&1 | sed -n 's/^FAILED \([^ ]*\).*/\1/p'
}

cpp_build() {
  cd "${repository_root}"
  cmake --build --preset coverage --target "$1" >/dev/null 2>&1
}

cpp_suite() {
  # $1 ctest regular expression. Prints failing test names, one per line.
  cd "${repository_root}"
  if [ -n "$1" ]; then
    ctest --preset coverage -R "$1" 2>&1 |
      sed -n 's/^[[:space:]]*[0-9]* - \(.*\) (Failed)$/\1/p'
  else
    ctest --preset coverage 2>&1 |
      sed -n 's/^[[:space:]]*[0-9]* - \(.*\) (Failed)$/\1/p'
  fi
}

cpp_coverage() {
  # $1 source path. Needs lcov or gcovr; reports the line rate for that path.
  cd "${repository_root}"
  if command -v gcovr >/dev/null 2>&1; then
    gcovr --root "${repository_root}" --filter "$1" --print-summary 2>/dev/null |
      sed -n 's/^lines: \(.*\)$/lines: \1/p' | tail -n 1
  elif command -v lcov >/dev/null 2>&1; then
    lcov --capture --directory build/coverage --output-file /dev/stdout \
      --quiet 2>/dev/null | lcov --extract /dev/stdin "*/$1/*" \
      --output-file /dev/null --quiet 2>&1 | sed -n 's/.*lines\.*: \(.*\)$/\1/p'
  else
    echo "unavailable: install gcovr or lcov for the C++ coverage tier"
  fi
}

tier=""
lang=""
source_path=""
tests_path=""
target=""
ctest_filter=""
drop_node=""
inject_spec=""
replacement=""
keep_edit="no"

[ $# -ge 1 ] || {
  usage
  exit 2
}

case "$1" in
  t1 | t2)
    tier="$1"
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *) fail "first argument must be t1, t2, or --help" ;;
esac

while [ $# -gt 0 ]; do
  case "$1" in
    --lang)
      lang="$2"
      shift 2
      ;;
    --source)
      source_path="$2"
      shift 2
      ;;
    --tests)
      tests_path="$2"
      shift 2
      ;;
    --target)
      target="$2"
      shift 2
      ;;
    --ctest)
      ctest_filter="$2"
      shift 2
      ;;
    --drop)
      drop_node="$2"
      shift 2
      ;;
    --inject)
      inject_spec="$2"
      shift 2
      ;;
    --with)
      replacement="$2"
      shift 2
      ;;
    --keep)
      keep_edit="yes"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *) fail "unknown option: $1" ;;
  esac
done

case "${lang}" in
  python | cpp) ;;
  *) fail "--lang must be python or cpp" ;;
esac

require_clean_tree
trap 'restore_edit' EXIT HUP INT TERM

baseline=$(git -C "${repository_root}" rev-parse HEAD)

if [ "${tier}" = "t1" ]; then
  [ -n "${source_path}" ] || fail "t1 needs --source"
  echo "=== SpecAudit probe: T1 coverage delta ==="
  echo "baseline commit : ${baseline}"
  echo "source          : ${source_path}"
  if [ "${lang}" = "python" ]; then
    [ -n "${tests_path}" ] || fail "t1 --lang python needs --tests"
    [ -n "${drop_node}" ] || fail "t1 --lang python needs --drop"
    echo "suite           : ${tests_path}"
    echo "dropped         : ${drop_node}"
    echo "with the test   : $(python_coverage "${source_path}" \
      "${tests_path}" '')"
    echo "without it      : $(python_coverage "${source_path}" \
      "${tests_path}" "${drop_node}")"
  else
    [ -n "${target}" ] || fail "t1 --lang cpp needs --target"
    cpp_build "${target}"
    cpp_suite "${ctest_filter}" >/dev/null || true
    echo "coverage        : $(cpp_coverage "${source_path}")"
    echo "note            : re-run after removing the assertion by hand and"
    echo "                  compare. The C++ tier does not deselect for you."
  fi
  echo "reminder        : equal coverage is evidence of redundancy, never"
  echo "                  proof. Escalate to t2 before deleting anything."
  exit 0
fi

[ -n "${inject_spec}" ] || fail "t2 needs --inject FILE:LINE"
inject_file=${inject_spec%:*}
inject_line=${inject_spec##*:}
[ "${inject_file}" != "${inject_spec}" ] || fail "--inject wants FILE:LINE"

original=$(sed -n "${inject_line}p" "${repository_root}/${inject_file}")
overwrite_line "${inject_file}" "${inject_line}" "${replacement}"

if [ "${lang}" = "python" ]; then
  [ -n "${tests_path}" ] || fail "t2 --lang python needs --tests"
  failures=$(python_suite "${tests_path}")
  suite_label="${tests_path}"
else
  [ -n "${target}" ] || fail "t2 --lang cpp needs --target"
  if cpp_build "${target}"; then
    failures=$(cpp_suite "${ctest_filter}")
  else
    failures="<build failed: the fault does not compile>"
  fi
  suite_label="${target}"
fi

count=$(printf '%s\n' "${failures}" | grep -c . || true)

echo "=== SpecAudit probe: T2 fault injection ==="
echo "baseline commit : ${baseline}"
echo "injected        : ${inject_file}:${inject_line}"
echo "original line   : ${original}"
echo "replaced with   : ${replacement}"
echo "suite           : ${suite_label}"
echo "failing tests   : ${count}"
if [ "${count}" -gt 0 ]; then
  printf '%s\n' "${failures}" | sed 's/^/  - /'
  echo "reading         : another test catches this fault. The candidate is"
  echo "                  redundant unless it is one of the tests listed."
else
  echo "reading         : nothing noticed. Check the spec ledger. If no rung"
  echo "                  1 to 3 promise makes this fault visible, the"
  echo "                  behaviour is contract-free and the code you just"
  echo "                  broke is a removal candidate."
fi
