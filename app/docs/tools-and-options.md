# Tools and options

Every tool in LTX Desktop, with the settings each one offers. For prompt rewriting, see [Prompt enhancement](#prompt-enhancement-local).

## Tools

| Section | Tool | Input | What it does |
| --- | --- | --- | --- |
| Workflows | Text to Video | Prompt | Generates a video with matching audio from a text prompt |
| Workflows | Image to Video | Start frame, optional end frame | Animates a still image, or interpolates between two frames |
| Workflows | Audio to Video | Audio (upload or microphone), optional start frame | Generates visuals synchronized to an audio track |
| Post production | Retake | Video | Regenerates a selected part of a clip (2 to 10 seconds) |
| Post production | Extend | Video | Adds 4 to 12 seconds before or after a clip |
| Post production | Day to Night, Deblur, Colorization, Clean Plate, Decompression, Water Simulation | Video | IC-LoRA effects that keep the motion and framing of the source |
| VFX | Layout to Render | Video and look image | IC-LoRA tool that renders a layout in the style of a reference image |
| VFX | Restore | Video, optional look image | IC-LoRA tool that restores old footage. A look image is optional |
| Styles | Cozy Felt, Claymation, Fantasy Painterly, Paper Cut-Out | Prompt | Text-to-video styles |
| Styles | Cinemagraph | Start frame | Image-to-video motion |
| Camera & motion | Drone Flythrough, OpenWheel T-Cam | Prompt | Text-to-video camera motion |
| Camera & motion | Jib Up, Jib Down, Dolly In, Dolly Out, VBVR | Start frame | Image-to-video camera moves |
| Camera & motion | Transition | Start frame and end frame | Generates a transition between two images |

## Generation options

The forms share one set of video settings. The app lists only the options that your downloaded models and your hardware can run. Longer clips and higher resolutions are greyed out, with the reason, on machines that cannot run them (see [`hardware-compatibility.md`](hardware-compatibility.md)).

### Text to Video, Image to Video, and LoRA tools

| Setting | Options |
| --- | --- |
| Model | **LTX 2.5 Fast** (default) and **LTX 2.3 Fast**. Only downloaded models are listed |
| Resolution | 270p, 360p, 540p (default), 720p, 1080p |
| Aspect ratio | 16:9 (default for Text to Video), 9:16, 21:9, 3:2, 4:3, 1:1, 4:5. Image to Video and the image-to-video LoRA tools also offer **Auto** (the default), which follows the start frame |
| Duration | 2, 3, 4, 5, 6, 8 (default), 10, or 20 seconds. At 1080p: 2, 3, 4, 5, or 10 seconds |
| Frame rate | 24 (default), 25, 48, or 50 fps |
| Seed | A random seed per run, or a fixed seed. The lock keeps the same seed between runs |
| Prompt enhancement | Automatic, or manual with the **Enhance** button |

The models were trained on 16:9 and 9:16. Other aspect ratios run, but can give lower quality, and the app shows a warning.

LoRA tools add **Style strength** (0 to 3) and, when a LoRA offers more than one adapter, a **Variant** menu. The image-to-video LoRA tools also show **Start Frame**, and Transition also shows **End Frame**.

### Audio to Video

| Setting | Options |
| --- | --- |
| Audio | Upload a file or record from the microphone (Windows and macOS). Audio must be at least 2 seconds. Longer audio can be trimmed with the **Trim** control |
| Start frame | Optional. Anchors the first frame of the video |
| Resolution | 270p, 360p, 540p (default), 720p, 1080p |
| Aspect ratio | The same ratios as above, plus **Auto** (default), which follows the start frame |
| Duration | Set by the audio. The maximum depends on the resolution and your hardware (up to 20 seconds, and up to 10 seconds at 1080p) |
| Frame rate | 24 fps |

### Retake and Extend

| Setting | Options |
| --- | --- |
| Video | Upload a file or pick one from the Assets library. An optional **Trim** control cuts a long source before you start. The source can be any length |
| Retake selection | 2 to 10 seconds, chosen on the clip timeline |
| Extend duration | 4, 6, 8, 10, or 12 seconds, added **before** or **after** the clip |
| Resolution | **Original** (up to 1080p), or a lower resolution. Retake and Extend never upscale |
| Model | Any downloaded model that supports the tool |

### IC-LoRA tools

| Setting | Options |
| --- | --- |
| Reference video | Upload a file or pick one from the Assets library. The input is limited to 10 seconds (240 frames) |
| Look image | A reference still that sets the style of the result. Layout to Render needs one. Restore accepts one, but does not need it |
| Resolution | 270p to 1080p. The app offers only the resolutions that the source video is large enough for. Layout to Render also offers sizes up to twice the source, so a 540p clip can render at 1080p |
| Frame rate | 24, 25, 48, or 50 fps. **Original** keeps the closest supported rate |
| Audio | **Off**, **Source** (keeps the original soundtrack), or **Generated** |
| Style strength | 0 to 2 |
| Variant | Selects the adapter variant when the LoRA has more than one |

## Prompt enhancement (local)

LTX models work best with long, detailed prompts that describe the scene, the motion, and (for LTX 2.5) the sound. Prompt enhancement rewrites a short prompt into that style before the model sees it. This section covers local generation. The API has its own switch in **Settings > General > Generation** (see [API keys, cost, and privacy](api-keys-and-privacy.md)).

The **Auto-enhance on local generation** switch is in **Settings > General > Generation**. It is **ON** by default. It applies to Text to Video, Image to Video, Audio to Video, and the LoRA and IC-LoRA tools. Retake and Extend never rewrite the prompt.

| | Auto-enhance **ON** | Auto-enhance **OFF** |
| --- | --- | --- |
| When the prompt is rewritten | Automatically, when you start a local generation | Only when you click **Enhance** |
| Prompt the model receives | The rewritten prompt | The prompt exactly as typed (or as you enhanced it) |
| **Enhance** button | Hidden | Shown next to the prompt field |

**With auto-enhance ON**, you do not need to do anything. The app rewrites the prompt with the local prompt enhancer just before it loads the video model. Notes:

- The local prompt enhancer is a Gemma model. Download it in **Settings > Models** (see [Managing models](models-and-loras.md#managing-models)). Without it, the app cannot rewrite locally and uses the prompt as typed.
- Image to Video and Audio to Video use the start frame as well as the text, so the rewrite matches what is in the image.
- LoRA and IC-LoRA tools get a rewrite that keeps the LoRA's trigger phrase. The trigger phrase is checked again after the rewrite.
- If the rewrite fails or returns nothing, the generation still runs with the original prompt.
- If you used **Enhance** on this prompt and did not edit the result, the app does not rewrite it a second time. If you edit the text, the app treats it as a typed prompt again.
- The seed control still works. With the seed locked, or an explicit seed, the rewrite uses the same seed. The same seed and prompt then give the same video. With an unlocked seed, each run can produce a different rewrite.

**With auto-enhance OFF**, the prompt goes to the model as typed. This gives you full control over the exact words. The **Enhance** button appears next to the prompt field, and only when auto-enhance is OFF. Notes:

- **Enhance** needs a prompt or a start frame.
- Use the **Enhance via** menu to choose the provider: **Local Gemma** (needs the local prompt enhancer) or **Gemini API** (needs a Gemini API key).
- The rewritten text replaces the prompt in the field, and the earlier text stays in the prompt history. You can edit the result before you generate. Generation uses the text as it stands.
- Click **Enhance** again to get a different rewrite.
