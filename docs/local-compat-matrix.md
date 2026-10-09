# Local Explore compatibility matrix

Generated from `decide_local_generation_mode` and `decide_video_job`.
A machine that cannot run a Fast cell is not offered that cell.
Regenerate: `cd backend && uv run python -m runtime_config.local_compat_matrix`.

| Tool | CUDA unsupported | CUDA stream | CUDA full | Darwin unsupported | Darwin stream | Intel Mac / no GPU |
| --- | --- | --- | --- | --- | --- | --- |
| T2V | No | Yes | Yes | No | Yes | No |
| I2V | No | Yes | Yes | No | Yes | No |
| A2V | No | Yes | Yes | No | Yes | No |
| LoRAs | No | Yes | Yes | No | Yes | No |
| IC-LoRA | No | Yes | Yes | No | Yes | No |
| Retake | No | Yes | Yes | No | Yes | No |
| Extend | No | Yes | Yes | No | Yes | No |
| local text encoder | No | Yes | Yes | No | Yes | No |
| 2.5 download | No | Yes | Yes | No | Yes | No |

Floors: CUDA integer **15** GiB VRAM (`total_memory // 1024**3`);
Darwin **32** GiB total RAM; ~72 GB disk for the 2.5 Fast core pack
(kitchen-sink installs with 2.3, image models, and processors are larger).

CUDA stream (16 GB) still runs T2V/I2V/A2V; the Fast picker hides 720p/20s
and 1080p/10s because `decide_video_job` would reject them. CUDA full (31 GB)
keeps those cells (they stream). Darwin stream keeps the static Fast ceiling.

Footnote: Darwin IC-LoRA duration is unbounded past the 540p spatial cap
(no token 422). Do not invent a CUDA-curve ceiling for Darwin.
