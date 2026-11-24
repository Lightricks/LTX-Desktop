# VideoSeal marking

Opt-in post-hoc VideoSeal on SDR H.264 encode and AudioSeal on stereo PCM. These are **not** cryptographic signatures; `detected` is each marker's own call (VideoSeal: bit-test `log10(p) < -6`; AudioSeal: presence score). `bit_accuracy` / `p_value` are the binomial test of decoded bits vs the baked payload.

## Install

```bash
uv sync --extra videoseal --extra audioseal
# or
pip install 'ltx-pipelines[videoseal,audioseal]'
```

PyPI `videoseal` 1.0.1 predates the PixelSeal card, so the extra installs videoseal from git. AudioSeal is PyPI `audioseal` 0.2+.

The library does not download weights. Pass a local `path/to/marking` pack directory on `--marker-root-path`. Expected files (all required):

- `pixelseal.pth`
- `pixelseal.yaml`
- `videoseal_attenuation.yaml`
- `audioseal_generator.pth`
- `audioseal_detector.pth`

## CLI

Video-generation and editing CLIs, HDR IC-LoRA, and text-to-audio (`t2a_one_stage`) accept `--marker-root-path DIR`. Omit the flag for unchanged pixels / PCM (`markers_from_args` returns `(None, None)`). HDR encode skips VideoSeal (logs that video marking is unavailable with HDR) and still stamps AudioSeal PCM before AAC/HLG mux. A2Vid two-stage stamps video only (source audio is muxed unmarked). Text-to-audio uses the same flag and only the audio marker (the pack still loads both).

```bash
python -m ltx_pipelines.distilled \
    ... \
    --marker-root-path path/to/marking \
    --output-path out.mp4

python -m ltx_pipelines.t2a_one_stage \
    ... \
    --marker-root-path path/to/marking \
    --output-path out.wav
```

Detect from Python with a loaded marker (there is no inspect CLI):

```python
import torch
from ltx_pipelines.marking import create_markers, detect_audio_mark, detect_video_mark

video_marker, audio_marker = create_markers("path/to/marking", torch.device("cuda"))
print(detect_video_mark(video_marker, "out.mp4"))
print(detect_audio_mark(audio_marker, "out.mp4"))
```

## Library

`encode_video(..., video_marker=None, audio_marker=None)` leaves pixels and PCM unchanged. Pass resident markers; encode does not load weights. HDR callers omit `video_marker`. A2Vid callers omit `audio_marker`. Each yielded `Audio` is stamped independently (encode does not stitch). `encode_audio(..., audio_marker=None)` is the WAV path.

From a pack (`create` loads, `eval()`s, drops grad, and moves to `device`):

```python
import torch
from ltx_pipelines.marking import create_markers
from ltx_pipelines.utils.media_io import encode_video

device = torch.device("cuda")
video_marker, audio_marker = create_markers("path/to/marking", device)
with torch.inference_mode():
    encode_video(..., video_marker=video_marker, audio_marker=audio_marker)
```

Without a pack, construct the seal classes from file paths:

```python
import torch
from ltx_pipelines.marking.audio_seal import AudioSealMarker
from ltx_pipelines.marking.video_seal import VideoSealMarker

device = torch.device("cuda")
video_marker = VideoSealMarker.create(
    "path/to/marking/pixelseal.pth",
    card="path/to/marking/pixelseal.yaml",
    attenuation_config="path/to/marking/videoseal_attenuation.yaml",
    device=device,
)
audio_marker = AudioSealMarker.create(
    "path/to/marking/audioseal_generator.pth",
    "path/to/marking/audioseal_detector.pth",
    device=device,
)

with torch.inference_mode():
    encode_video(..., video_marker=video_marker, audio_marker=audio_marker)
```

`VideoMarker.embed` stamps one `[F, H, W, C]` RGB tensor in `[0, 1]`. `VideoMarker.detect` takes an iterator of `[H, W, C]` frames in the same domain. `AudioMarker.embed` / `AudioMarker.detect` take `Audio`. File I/O is `detect_video_mark` / `detect_audio_mark` in `ltx_pipelines.marking`.

Pack load uses the local card YAML (not packaged `videoseal/cards/`). Stereo is stamped as a batch of two mono channels (no downmix). Native 24/48 kHz; AudioSeal 0.2 does not resample.

Public API in `ltx_pipelines.marking` is the `VideoMarker` / `AudioMarker` protocols, `create_markers`, `MarkDetectResult`, and detect helpers. `VideoSealMarker.create` / `AudioSealMarker.create` live in `ltx_pipelines.marking.video_seal` and `ltx_pipelines.marking.audio_seal` (not re-exported, so importing `ltx_pipelines.marking` does not require extras) and return those protocols. Filename constants for the pack (`PIXELSEAL_CHECKPOINT_NAME`, …) live on the root helper.
