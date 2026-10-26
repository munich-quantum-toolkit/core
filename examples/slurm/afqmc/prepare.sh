#!/bin/sh
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Run from the repository root. Existing runtime directories are never overwritten.
set -eu
runtime=${1:-build/slurm-afqmc}
sh examples/slurm/prepare.sh "$runtime"
mkdir "$runtime/upstream"
curl -fLsS https://github.com/amazon-braket/amazon-braket-examples/archive/16cd791da7c3ec7e104851eb3bc502d00be1c1c3.tar.gz -o "$runtime/upstream.tar.gz"
printf '%s  %s\n' cdfa556e8dde2bf94e144afbf0adfb3522cbf72bc726f945f5bf085075db8966 "$runtime/upstream.tar.gz" | sha256sum -c -
tar -xzf "$runtime/upstream.tar.gz" -C "$runtime/upstream" --strip-components=4 \
    amazon-braket-examples-16cd791da7c3ec7e104851eb3bc502d00be1c1c3/examples/hybrid_quantum_algorithms/Quantum_Monte_Carlo_Chemistry
rm "$runtime/upstream.tar.gz"
python3 - "$runtime" <<'PY'
import json
import sys
from pathlib import Path
runtime = Path(sys.argv[1])
resources = (runtime / "resources.yaml").read_text()
template = resources[resources.index("- resource:"):resources.index("- resource: mqt.sc.default")]
for device in ("iqm.emerald.mock", "amazon.braket.sv1"):
    resources += template.replace("mqt.ddsim.default", device).replace("count: 2", "count: 1")
(runtime / "resources.yaml").write_text(resources)
config = (runtime / "slurm.conf").read_text().replace("AccountingStorageTRES=", "AccountingStorageTRES=license/iqm.emerald.mock,license/amazon.braket.sv1,")
(runtime / "slurm.conf").write_text(config)
(runtime / "qdmi.json").write_text(json.dumps({"schema-version": 1, "qdmi": {"devices": [
    {"id": "iqm.emerald.mock", "session": {"base-url": "http://iqm:8080"}},
]}}))
(runtime / "qdmi.env").write_text("AWS_ACCESS_KEY_ID=local-fixture\nAWS_SECRET_ACCESS_KEY=local-fixture\nAWS_EC2_METADATA_DISABLED=true\nAWS_ENDPOINT_URL=http://braket:8080\nAWS_DEFAULT_REGION=us-east-1\n")
PY
