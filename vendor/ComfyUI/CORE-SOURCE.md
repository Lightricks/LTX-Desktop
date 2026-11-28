# ComfyUI core

The Windows AMD installer includes the complete, unmodified official source archive for ComfyUI v0.38.0, released under GPL-3.0. Its license and source notices are also inside the ZIP. The archive contains the complete corresponding source for the private worker. LTX's provisioning and startup code lives in `app/backend/services/amd_comfy_provision.py` and `amd_comfy_bootstrap.py`.

| Property | Value |
| --- | --- |
| Release | [ComfyUI v0.38.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.38.0) |
| Commit | `6b747c0428c343e1417219641db93a4fb7cb69ae` |
| Archive | [Upstream ZIP by commit](https://codeload.github.com/Comfy-Org/ComfyUI/zip/6b747c0428c343e1417219641db93a4fb7cb69ae) |
| Bytes | 13,285,248 |
| SHA-256 | `4a144a553c77397c81f6067f3f8c8ab9dffd42889af48fb35fec2c64a71c75b2` |

ComfyUI's supported LTX models, text encoder and ConvRot quantization paths are in `comfy/supported_models.py`, `comfy/sd.py` and `comfy/ops.py`. The application's two LTX graphs use core nodes, and the optional Kandinsky workflow uses the separate Kandinsky Lab node pack. ComfyUI can also run independently using [its published installation instructions](https://github.com/Comfy-Org/ComfyUI/blob/6b747c0428c343e1417219641db93a4fb7cb69ae/README.md).
