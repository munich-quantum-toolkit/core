-- Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
-- Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
-- All rights reserved.
--
-- SPDX-License-Identifier: MIT
--
-- Licensed under the MIT License

-- Run with Lua 5.4; the final check invokes the real process deadline.
slurm = {SUCCESS = 0, ERROR = -1, ESLURM_HRES_MAINTENANCE_DISABLE = 17}
dofile(arg[1])
local layers = {{disable_layer = false}, {disable_layer = false}}
local context = {name = "device'; touch /tmp/hres-shell-injection; echo '", config = {probe_timeout = 1}}
assert(hres_create_context(context, layers) == 17)
local execute = os.execute
for _, status in ipairs({true, 0, false, 1, 124}) do
    os.execute = function(command)
        assert(command:find("/usr/bin/timeout --kill-after=2s 1s", 1, true))
        return status
    end
    assert(hres_maintenance(context, layers) == 0)
    for _, layer in ipairs(layers) do
        assert(layer.disable_layer == not (status == true or status == 0))
    end
end
for _, invalid in ipairs({"garbage", 0, -1, 1.5, 3601, "1; touch /tmp/hres-shell-injection"}) do
    context.config.probe_timeout = invalid
    os.execute = function() error("Invalid timeout reached a shell") end
    assert(hres_maintenance(context, layers) == -1)
    for _, layer in ipairs(layers) do assert(layer.disable_layer) end
end
os.execute = execute
context.config.probe_timeout = 1
-- The harness mounts a sleeping probe at the configured executable path.
assert(hres_maintenance(context, layers) == 0)
for _, layer in ipairs(layers) do assert(layer.disable_layer) end
assert(not io.open('/tmp/hres-shell-injection'))
