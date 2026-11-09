-- Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
-- Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
-- All rights reserved.
--
-- SPDX-License-Identifier: MIT
--
-- Licensed under the MIT License

-- Lua 5.4 self-check; no Slurm installation or provider credentials needed.
slurm = {SUCCESS = 0, ESLURM_HRES_MAINTENANCE_DISABLE = 17}
dofile(arg[1])
local layers = {{disable_layer = false}, {disable_layer = true}}
local context = {name = "device'quoted"}
assert(hres_create_context(context, layers) == 17)
for _, status in ipairs({0, 1, 124}) do
    os.execute = function(command)
        assert(command == "/usr/bin/timeout --kill-after=2s 10s /usr/local/libexec/mqt-qdmi-probe " ..
            "'device'\\''quoted' </dev/null >/dev/null 2>&1")
        if status == 0 then return true, "exit", 0 end
        return nil, "exit", status
    end
    assert(hres_maintenance(context, layers) == 0)
    for _, layer in ipairs(layers) do assert(layer.disable_layer == (status ~= 0)) end
end
