-- Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
-- Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
-- All rights reserved.
--
-- SPDX-License-Identifier: MIT
--
-- Licensed under the MIT License

local function quote(value)
    return "'" .. (tostring(value):gsub("'", "'\\''")) .. "'"
end

function hres_create_context(ctx, layers)
    -- Admission starts closed on startup and reconfiguration.
    return slurm.ESLURM_HRES_MAINTENANCE_DISABLE
end

function hres_maintenance(ctx, layers)
    for _, layer in ipairs(layers) do layer.disable_layer = true end
    local deadline = tonumber(ctx.config.probe_timeout)
    if not deadline or deadline < 1 or deadline > 3600 or deadline % 1 ~= 0 then
        return slurm.ERROR
    end
    -- Slurm's maintenance_timeout only logs slow calls; timeout enforces the bound.
    local command = string.format(
        "/usr/bin/timeout --kill-after=2s %ds /usr/local/libexec/mqt-qdmi-probe %s </dev/null >/dev/null 2>&1",
        deadline, quote(ctx.name))
    local rc = os.execute(command)
    local available = (rc == true or rc == 0)
    for _, layer in ipairs(layers) do layer.disable_layer = not available end
    return slurm.SUCCESS
end

-- Remote sessions belong to the application; HRES accounts for cluster capacity.
local function success() return slurm.SUCCESS end
hres_destroy_context = success
hres_controller_job_prerun, hres_controller_job_revoke = success, success
hres_controller_job_epilog = success
hres_node_job_prolog, hres_node_job_epilog = success, success
hres_node_task_prolog, hres_node_task_epilog = success, success
hres_node_user_task_prolog = success
