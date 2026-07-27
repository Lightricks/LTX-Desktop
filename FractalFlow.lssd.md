<!--
Canonical-For: .
Status: ACTIVE
-->

# LTX Desktop — Runtime Selection and Local Accelerator Contract

This contract governs capability-aware local execution, memory isolation,
cross-product Metal coordination, telemetry, and the user-visible performance
surface. Code, generated API bindings, package builds, and tests are implementation
evidence; this document owns the behavioral invariants.

```lssd
model ltx_desktop.runtime@1.0.0 {
  namespace: "ltx_desktop"
  revision: "2026-07-26"
  status: observed
  confidence: 0.95
}

system ltx_desktop.runtime {
  label: "LTX Desktop Electron local and cloud generation runtime"
  status: observed
  confidence: 0.95
  contains: [
    @ltx_desktop.runtime_selection,
    @ltx_desktop.mlx_sidecar,
    @ltx_desktop.local_metal_lease,
    @ltx_desktop.runtime_telemetry,
    @ltx_desktop.performance_ui
  ]
}

module ltx_desktop.runtime_selection {
  label: "Capability-aware Fast video engine selection"
  status: observed
  confidence: 0.95
}

module ltx_desktop.mlx_sidecar {
  label: "Per-job isolated MLX Fast T2V and I2V process"
  status: observed
  confidence: 0.95
}

module ltx_desktop.local_metal_lease {
  label: "Cross-product advisory Metal accelerator lease"
  status: observed
  confidence: 0.99
}

module ltx_desktop.runtime_telemetry {
  label: "Live process, system, MPS, MLX child, and lease telemetry"
  status: observed
  confidence: 0.95
}

module ltx_desktop.performance_ui {
  label: "React settings performance and provenance surface"
  status: observed
  confidence: 0.95
}

truth ltx_desktop.capability_engine {
  label: "Engine selected for one generation capability"
  lifetime: task
  authority: { mode: capability, ref: @ltx_desktop.runtime_selection }
  visibility: [public]
  status: observed
  confidence: 0.95
  summary: "Auto may use MLX only for Fast T2V/I2V with local text encoding, a compatible pinned sidecar, and cached BF16 weights; prepared embeddings, A2V, Retake, Extend, IC-LoRA, and image generation remain on feature-complete Torch or cloud paths."
}

truth ltx_desktop.runtime_provenance {
  label: "Actual and compatibility-target runtime provenance"
  lifetime: system
  authority: { mode: derived, ref: @ltx_desktop.runtime_selection }
  visibility: [public]
  status: observed
  confidence: 0.95
}

truth ltx_desktop.mlx_job_profile {
  label: "Prompt-safe crash-resilient per-job MLX JSONL profile"
  lifetime: archive
  authority: { mode: derived, ref: @ltx_desktop.mlx_sidecar }
  visibility: [restricted]
  status: observed
  confidence: 0.99
  summary: "Each isolated child writes flushed phase, allocator, terminal status, and runtime identity evidence beside outputs; the API retains the last parsed snapshot after process exit."
}

truth ltx_desktop.local_metal_owner {
  label: "Diagnostic owner metadata for the shared Metal lease"
  lifetime: task
  authority: { mode: derived, ref: @ltx_desktop.local_metal_lease }
  visibility: [public]
  status: observed
  confidence: 0.99
  summary: "fcntl.flock is authoritative; JSON file content is diagnostic and may be stale after a crash."
}

policy ltx_desktop.no_silent_capability_loss on @ltx_desktop.runtime_selection {
  effect: require
  status: confirmed
  confidence: 1.0
  rule: "Auto selection must route any request not proven equivalent on MLX to Torch or cloud and expose the reason."
}

policy ltx_desktop.quality_default on @ltx_desktop.runtime_selection {
  effect: require
  status: confirmed
  confidence: 1.0
  rule: "MLX BF16 is the only automatic MLX quality tier; q8 is explicit expert-only, visibly warned, and never auto-selected."
}

policy ltx_desktop.no_hidden_model_download on @ltx_desktop.runtime_selection {
  effect: require
  status: confirmed
  confidence: 1.0
  rule: "Auto mode must not download MLX weights; an uncached model falls back to Torch with a visible reason."
}

policy ltx_desktop.isolated_mlx_lifetime on @ltx_desktop.mlx_sidecar {
  effect: require
  status: confirmed
  confidence: 1.0
  rule: "Every MLX generation runs in a fresh child process group; the FastAPI and Electron processes never import or retain MLX model state, process exit is the allocator teardown boundary, and cancellation kills the entire child process group."
}

policy ltx_desktop.exact_mlx_runtime_identity on @ltx_desktop.runtime_selection {
  effect: require
  status: confirmed
  confidence: 1.0
  rule: "Discover LTX_MLX_PYTHON first, then ~/video-models/ltx-2-mlx/.venv/bin/python, then a console entrypoint; query ltx_pipelines_mlx.utils.runtime_info without loading a model, and admit MLX only when runtime version and source revision exactly match the compatibility pin and the source checkout is not dirty."
}

policy ltx_desktop.durable_mlx_evidence on @ltx_desktop.mlx_sidecar {
  effect: require
  status: confirmed
  confidence: 1.0
  rule: "Every MLX child receives a unique --profile-json path; complete flushed records are ingested without making telemetry fatal to inference while malformed tails, invalid timestamps, and non-finite allocator values are ignored, terminal allocator peak/active/cache, last phase, status, path, timestamp, and prompt-safe runtime identity remain visible after child exit, cancellation records a synthetic terminal snapshot if SIGKILL prevents a final flush, and strict HD qualification samples the terminal API snapshot after child completion."
}

policy ltx_desktop.aspect_aware_fast_dimensions on @ltx_desktop.mlx_sidecar {
  effect: require
  status: confirmed
  confidence: 1.0
  rule: "Local Fast T2V/I2V resolves one aspect-aware, two-stage 64-pixel-grid size before image preparation or inference; 540p uses budget-bounded 896x512 (or 512x896 portrait) instead of independently rounding 960x544 to distorted 960x512, the completion API and generated client bindings expose the resolved pair, the MLX adapter rejects off-grid dimensions before launch, and strict QA requires reported resolution to equal ffprobe output."
}

policy ltx_desktop.shared_metal_authority on @ltx_desktop.local_metal_lease {
  effect: require
  status: confirmed
  confidence: 1.0
  rule: "Local Metal model loading and generation hold an exclusive nonblocking fcntl.flock at ~/Library/Application Support/LTX Shared/local-metal.lock from before model load through cleanup; cloud/API work bypasses it."
}

policy ltx_desktop.cancellable_lease_wait on @ltx_desktop.local_metal_lease {
  effect: require
  status: confirmed
  confidence: 1.0
  rule: "Lease acquisition polls every 250 milliseconds, checks cancellation every poll, and surfaces wait duration plus untrusted owner diagnostics."
}

obligation ltx_desktop.visible_runtime_truth on @ltx_desktop.performance_ui {
  status: confirmed
  confidence: 0.95
  rule: "The performance UI exposes selection preference, selected engine and reason, eager or low-RAM mode, automatic tiling, live or last-job allocator memory, MLX profile status/phase/path/runtime identity, lease status, capability routing, model tier warnings, and actual plus target provenance."
}

flow ltx_desktop.fast_generation {
  label: "Capability-aware local Fast generation"
  status: observed
  confidence: 0.95
  entry: requested
  terminal: complete

  state requested
  state routed
  state lease_wait
  state model_ready
  state generating
  state cleaning
  state complete
  state cancelled
  state failed

  transition requested -> routed on capabilities_and_runtime_evaluated
  transition routed -> lease_wait on local_engine_selected
  transition lease_wait -> cancelled on cancellation_requested
  transition lease_wait -> model_ready on shared_metal_lease_acquired
  transition model_ready -> generating on generation_started
  transition generating -> cancelled on cancellation_kills_mlx_group_or_torch_observes_state
  transition generating -> cleaning on output_written
  transition cleaning -> complete on profile_ingested_allocator_cleanup_and_lease_release
  transition model_ready -> failed on model_load_failure
  transition generating -> failed on inference_failure
}

artifact ltx_desktop.runtime_policy_code {
  kind: "runtime selection policy"
  path: "backend/runtime_config/runtime_policy.py"
  status: observed
  confidence: 0.99
}

artifact ltx_desktop.mlx_runtime_discovery_code {
  kind: "portable exact-identity sidecar discovery"
  path: "backend/runtime_config/mlx_runtime.py"
  status: observed
  confidence: 0.99
}

artifact ltx_desktop.mlx_sidecar_code {
  kind: "isolated MLX adapter"
  path: "backend/services/fast_video_pipeline/mlx_fast_video_pipeline.py"
  status: observed
  confidence: 0.99
}

artifact ltx_desktop.fast_resolution_code {
  kind: "aspect-aware local Fast output grid"
  path: "backend/handlers/video_resolution.py"
  status: observed
  confidence: 0.99
}

artifact ltx_desktop.mlx_profile_code {
  kind: "durable per-job MLX profile ingestion"
  path: "backend/services/fast_video_pipeline/mlx_profile.py"
  status: observed
  confidence: 0.99
}

artifact ltx_desktop.local_metal_lease_code {
  kind: "cross-product lock implementation"
  path: "backend/services/local_metal_lease.py"
  status: observed
  confidence: 0.99
}

artifact ltx_desktop.runtime_api_code {
  kind: "runtime policy and telemetry API"
  path: "backend/handlers/runtime_policy_handler.py"
  status: observed
  confidence: 0.99
}

artifact ltx_desktop.performance_ui_code {
  kind: "runtime settings UI"
  path: "frontend/components/SettingsModal.tsx"
  status: observed
  confidence: 0.99
}

artifact ltx_desktop.package_manifest {
  kind: "Electron packaging contract that bundles backend Python sources"
  path: "electron-builder.yml"
  status: observed
  confidence: 0.99
}
```
