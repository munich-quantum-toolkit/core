#!/bin/sh

OUTPUT_DIR="tmp-output"

MQT_CORE_ROOT_DIR=..
MQT_BENCH_BENCHMARK_DIR="../../mqt-bench/generated_benchmarks/v3_qasm3"
MQT_BENCH_PATTERN="*.qasm"

mkdir -p "${OUTPUT_DIR}/broken"

benchmark_count="$(eza -1 ${MQT_BENCH_BENCHMARK_DIR}/${MQT_BENCH_PATTERN} | wc -l)"
i=0
success=0

for benchmark_path in ${MQT_BENCH_BENCHMARK_DIR}/${MQT_BENCH_PATTERN}; do
  i=$((i + 1))
  benchmark="$(basename "${benchmark_path}")"
  echo "${i}/${benchmark_count}: ${benchmark}"
  result="$("${MQT_CORE_ROOT_DIR}/build/mlir/tools/mqt-cc/mqt-cc" --mlir-timing --mlir-statistics "${benchmark_path}" 2>&1)"
  if [ "${?}" -eq 0 ]; then
    echo "${result}" | rg 'Total Execution Time' -A 8 >"${OUTPUT_DIR}/${benchmark}.timing"
    echo "${result}" | rg '\(S\) ' >"${OUTPUT_DIR}/${benchmark}.statistic"
    echo "${result}" | rg 'module \{' -A 99999999999999 >"${OUTPUT_DIR}/${benchmark}.mlir"
    echo "${result}" >"${OUTPUT_DIR}/${benchmark}.all"
    echo "SUCCESS"
    success=$((success + 1))
  else
    echo "${result}" >"${OUTPUT_DIR}/broken/${benchmark}.error"
    echo "FAILED"
  fi
done

echo "COMPLETED: ${success}/${i}"
