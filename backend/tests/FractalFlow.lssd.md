<!-- Canonical-For: backend/tests/test_performance_hd_matrix.py; Status: ACTIVE -->
# Backend Tests FractalFlow LSSD

```lssd
model ltx_desktop.backend_tests @ 0.1.0 {
  status: implemented
  confidence: 0.98
  summary: "Integration-first and pure-contract tests for the local FastAPI backend and its QA tooling."
}

artifact tests.performance_hd_matrix {
  path: "backend/tests/test_performance_hd_matrix.py"
  status: implemented
  confidence: 0.99
  summary: "No-model guards for HD matrix coverage, API-resolved versus ffprobe-actual dimension equality, complete macOS rusage_info_v2 ABI sizing, focused Metal GPU interval reference/PID parsing, truthful TeaCache/tiling scope, cross-product ordered ownership plus normalized live/release owner diagnostics and CPU-control-plane responsiveness, flock authority/release, and hard failure when authoritative telemetry is absent."
}

artifact tests.mlx_sidecar_isolation {
  path: "backend/tests/test_mlx_fast_video_pipeline.py"
  status: implemented
  confidence: 0.99
  summary: "Proves every MLX render starts a fresh process group with child-local memory settings, forwards BF16 precision plus low-RAM auto-tiling, rejects non-64-grid distilled dimensions before launch, and cancellation kills the group without retaining an active child."
}

artifact tests.fast_resolution_contract {
  path: "backend/tests/test_video_resolution.py"
  status: implemented
  confidence: 0.99
  summary: "Proves the 540p landscape and portrait Fast pairs are 64-grid-valid, stay under the nominal pixel budget, improve the requested 16:9 approximation, and reject unsupported catalog/aspect inputs."
}

artifact tests.mlx_runtime_discovery {
  path: "backend/tests/test_mlx_runtime.py"
  status: implemented
  confidence: 0.99
  summary: "Proves explicit Python precedence, sibling-runtime fallback, exact version and revision admission, and dirty-checkout rejection without loading a model."
}

artifact tests.mlx_profile_ingestion {
  path: "backend/tests/test_mlx_profile.py"
  status: implemented
  confidence: 0.99
  summary: "Proves flushed JSONL ingestion, incomplete-tail and non-finite-value tolerance, prompt-safe runtime identity filtering, allocator active/cache/peak retention after child exit, phase retention, and synthetic cancellation state without model work."
}

obligation tests.no_heavy_side_effects {
  status: implemented
  confidence: 0.99
  summary: "Default tests never load a model, start a Metal render, contact a provider, or mutate a running app queue."
}
```
