---
name: LTX-Desktop MCP 视频生成与编辑
description: 使用 LTX-Desktop MCP Server 进行 AI 视频生成、时间轴编辑和视频导出的完整指南
---

# LTX-Desktop MCP 视频生成与编辑

## 概述

LTX-Desktop 是一个基于 AI 的视频生成和编辑桌面应用。通过 MCP Server，AI 助手可以：
- **生成视频/图片**：调用后端 API（端口 8000）
- **编辑时间轴**：通过 Electron 桥（端口 8100）读写项目 JSON
- **导出成片**：调用 FFmpeg 合成导出

## 前置条件

1. LTX-Desktop 必须已启动（`pnpm dev`），确保后端（:8000）和桥（:8100）都在运行
2. MCP Server 脚本路径：`/home/ve/下载/LTX-Desktop/ltx_mcp_server.py`

---

## MCP 工具详细参数说明

### 一、视频生成工具

#### `generate_video` — 生成视频

从文本提示词生成视频，或基于输入图片/音频生成视频。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `prompt` | string | ✅ 必填 | — | 视频内容描述。不能为空，前后空格会被自动去除 |
| `resolution` | string | 选填 | `"512p"` | 生成分辨率 |
| `model` | string | 选填 | `"fast"` | 模型变体。`"fast"` = 快速蒸馏模型，`"pro"` = 高质量模型 |
| `duration` | string | 选填 | `"2"` | 视频时长（秒），传字符串 |
| `fps` | string | 选填 | `"24"` | 帧率，传字符串 |
| `camera_motion` | string | 选填 | `"none"` | 摄像机运动方式，见下表 |
| `negative_prompt` | string | 选填 | `""` | 负面提示词，描述要避免的内容 |
| `audio` | string | 选填 | `"false"` | 是否同时生成音频。`"true"` 或 `"false"` |
| `image_path` | string | 选填 | `null` | 输入图片的**本地绝对路径**，用于图生视频（image-to-video） |
| `audio_path` | string | 选填 | `null` | 输入音频的**本地绝对路径**，用于音生视频（audio-to-video） |
| `aspect_ratio` | string | 选填 | `"16:9"` | 画面比例。仅支持 `"16:9"` 或 `"9:16"` |

**`camera_motion` 可选值：**

| 值 | 说明 |
|---|---|
| `"none"` | 无运动（默认） |
| `"static"` | 完全静止镜头 |
| `"dolly_in"` | 推镜头（向前移动） |
| `"dolly_out"` | 拉镜头（向后移动） |
| `"dolly_left"` | 左移 |
| `"dolly_right"` | 右移 |
| `"jib_up"` | 镜头上升 |
| `"jib_down"` | 镜头下降 |
| `"focus_shift"` | 焦点转移 |

**返回值：**
```json
{ "status": "complete", "video_path": "/absolute/path/to/output.mp4" }
```

---

#### `generate_image` — 生成图片

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `prompt` | string | ✅ 必填 | — | 图片内容描述。不能为空 |
| `width` | int | 选填 | `1024` | 图片宽度（像素） |
| `height` | int | 选填 | `1024` | 图片高度（像素） |
| `num_steps` | int | 选填 | `4` | 扩散步数，越大质量越高但越慢 |
| `num_images` | int | 选填 | `1` | 生成图片数量 |

**返回值：**
```json
{ "status": "complete", "image_paths": ["/path/to/img1.png", "/path/to/img2.png"] }
```

---

#### `retake_video` — 视频局部重拍

对已有视频的某个时间段进行 AI 重新生成。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `video_path` | string | ✅ 必填 | — | 源视频的本地绝对路径 |
| `start_time` | float | ✅ 必填 | — | 重拍起始时间（秒），如 `1.5` |
| `duration` | float | ✅ 必填 | — | 重拍时长（秒），如 `2.0` |
| `prompt` | string | 选填 | `""` | 重拍内容的提示词，为空则保持相似风格 |
| `mode` | string | 选填 | `"replace_audio_and_video"` | 重拍模式，同时替换音视频 |

**返回值：**
```json
{ "status": "complete", "video_path": "/path/to/retake_output.mp4" }
```

---

#### `suggest_gap_prompt` — AI 建议补帧提示词

当时间轴上两段片段之间有空隙时，AI 根据上下文建议一个合适的提示词来生成过渡内容。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `before_prompt` | string | 选填 | `""` | 空隙前面那段片段的提示词 |
| `after_prompt` | string | 选填 | `""` | 空隙后面那段片段的提示词 |
| `gap_duration` | float | 选填 | `5` | 空隙时长（秒） |
| `mode` | string | 选填 | `"t2v"` | 生成模式。`"t2v"` = 文生视频 |
| `before_frame` | string | 选填 | `null` | 空隙前最后一帧图片的路径 |
| `after_frame` | string | 选填 | `null` | 空隙后第一帧图片的路径 |
| `input_image` | string | 选填 | `null` | 输入参考图片路径 |

**返回值：**
```json
{ "status": "success", "suggested_prompt": "A smooth camera pan transitioning from..." }
```

---

### 二、生成控制工具

#### `get_generation_progress` — 查询生成进度

无参数。实时查询当前视频/图片生成的进度。

**返回值：**
```json
{
  "status": "running",       // "idle" | "running" | "complete" | "cancelled" | "error"
  "phase": "denoising",      // 当前阶段描述
  "progress": 45,            // 总进度百分比 0-100
  "currentStep": 9,          // 当前扩散步数
  "totalSteps": 20           // 总扩散步数
}
```

---

#### `cancel_generation` — 取消生成

无参数。取消当前正在运行的生成任务。

**返回值：**
```json
{ "status": "cancelled", "id": "gen-xxx" }
```

---

### 三、模型管理工具

#### `get_health` — 健康检查

无参数。检查后端状态和 GPU 信息。

**返回值：**
```json
{
  "status": "ready",
  "models_loaded": true,
  "active_model": "fast",
  "gpu_info": { "name": "NVIDIA GB202", "vram": 131072, "vramUsed": 45000 },
  "sage_attention": true,
  "models_status": [
    { "id": "checkpoint", "name": "LTX Checkpoint", "loaded": true, "downloaded": true }
  ]
}
```

---

#### `get_models_status` — 模型下载状态

无参数。查看所有模型文件的下载状态和大小。

**返回值：**
```json
{
  "all_downloaded": false,
  "total_size_gb": 98.5,
  "downloaded_size_gb": 45.2,
  "models_path": "/home/ve/.local/share/LTXDesktop/models",
  "has_api_key": true,
  "models": [
    {
      "id": "checkpoint",
      "name": "LTX Video Checkpoint",
      "downloaded": true,
      "size": 12345678900,
      "expected_size": 12345678900,
      "required": true
    }
  ]
}
```

---

#### `download_models` — 触发模型下载

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `model_types` | list[string] | 选填 | `null`（全部下载） | 要下载的模型类型列表 |

**`model_types` 可选值：**

| 值 | 说明 |
|---|---|
| `"checkpoint"` | 主模型检查点（最大，~15GB） |
| `"upsampler"` | 超分辨率上采样器 |
| `"distilled_lora"` | 蒸馏 LoRA（用于 fast 模式加速） |
| `"ic_lora"` | 图像条件 LoRA |
| `"depth_processor"` | 深度估计处理器 |
| `"person_detector"` | 人体检测器 |
| `"pose_processor"` | 姿态估计处理器 |
| `"text_encoder"` | 文本编码器（~10GB，本地文本编码用） |
| `"zit"` | ZIT 图像修复模型 |

---

### 四、项目管理工具（Electron 桥）

#### `list_projects` — 列出所有项目

无参数。返回所有项目的摘要信息。

**返回值：** `Project[]` 数组（见下方 JSON 结构定义）

---

#### `get_project` — 获取项目完整数据

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `project_id` | string | ✅ 必填 | 项目 ID，格式如 `"project-1710000000000-abc123def"` |

**返回值：** 完整的 `Project` 对象 JSON（包含 assets、timelines、clips 等全部数据）

---

#### `update_project` — 更新项目

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `project_id` | string | ✅ 必填 | 项目 ID |
| `project_json` | string | ✅ 必填 | **完整的项目 JSON 字符串**。先用 `get_project` 读取，修改后传回 |

> ⚠️ **重要**：必须传完整的 Project JSON，不支持局部更新。先读后改再写！

---

#### `export_video` — 导出视频

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `project_id` | string | ✅ 必填 | — | 要导出的项目 ID |
| `output_path` | string | ✅ 必填 | — | 输出文件的本地绝对路径（如 `/home/ve/output.mp4`） |
| `width` | int | 选填 | `1920` | 输出视频宽度（像素） |
| `height` | int | 选填 | `1080` | 输出视频高度（像素） |
| `fps` | int | 选填 | `24` | 输出帧率 |
| `codec` | string | 选填 | `"h264"` | 编码器。`"h264"` / `"prores"` / `"vp9"` |
| `quality` | int | 选填 | `80` | 输出质量。h264: CRF 值(0-51, 越小越好)；prores: profile(0-3)；vp9: 比特率(MB) |

---

#### `import_asset` — 导入素材

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `project_id` | string | ✅ 必填 | 目标项目 ID |
| `file_path` | string | ✅ 必填 | 要导入的本地文件绝对路径（视频/图片/音频） |

**返回值：**
```json
{ "success": true, "path": "/copied/path/to/file.mp4", "url": "file:///copied/path/to/file.mp4" }
```

---

## 项目 JSON 完整结构定义

### Project（项目）

```json
{
  "id": "project-{timestamp}-{random9}",
  "name": "项目名称",
  "createdAt": 1710000000000,
  "updatedAt": 1710000000000,
  "thumbnail": "file:///path/to/thumb.png",
  "assets": [ /* Asset[] */ ],
  "timelines": [ /* Timeline[] */ ],
  "activeTimelineId": "timeline-xxx"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 唯一 ID，格式 `project-{timestamp}-{random9}` |
| `name` | string | 项目名称 |
| `createdAt` | number | 创建时间（毫秒时间戳） |
| `updatedAt` | number | 最后更新时间（毫秒时间戳） |
| `thumbnail` | string? | 项目缩略图 URL |
| `assets` | Asset[] | 项目中的所有素材 |
| `timelines` | Timeline[] | 时间轴列表（可多条） |
| `activeTimelineId` | string? | 当前激活的时间轴 ID |

---

### Asset（素材）

```json
{
  "id": "asset-{timestamp}-{random9}",
  "type": "video",
  "path": "/absolute/path/to/file.mp4",
  "url": "file:///absolute/path/to/file.mp4",
  "prompt": "生成时的提示词",
  "resolution": "512p",
  "duration": 5.0,
  "createdAt": 1710000000000,
  "thumbnail": "file:///path/to/thumb.png",
  "favorite": false,
  "takes": [],
  "activeTakeIndex": 0
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 唯一 ID |
| `type` | string | `"video"` / `"image"` / `"audio"` / `"adjustment"` |
| `path` | string | 文件在磁盘上的绝对路径 |
| `url` | string | `file://` 协议 URL |
| `prompt` | string? | AI 生成时的提示词 |
| `resolution` | string? | 生成分辨率 |
| `duration` | number? | 素材时长（秒），图片无此字段 |
| `createdAt` | number | 创建时间戳 |
| `thumbnail` | string? | 缩略图 URL |
| `favorite` | boolean | 是否收藏 |
| `takes` | AssetTake[] | 重拍/多版本历史 |
| `activeTakeIndex` | number? | 当前活跃版本索引 |

---

### Timeline（时间轴）

```json
{
  "id": "timeline-{timestamp}-{random9}",
  "name": "Timeline 1",
  "createdAt": 1710000000000,
  "tracks": [
    { "id": "track-v1", "name": "V1", "kind": "video", "muted": false, "locked": false },
    { "id": "track-v2", "name": "V2", "kind": "video", "muted": false, "locked": false },
    { "id": "track-v3", "name": "V3", "kind": "video", "muted": false, "locked": false },
    { "id": "track-a1", "name": "A1", "kind": "audio", "muted": false, "locked": false },
    { "id": "track-a2", "name": "A2", "kind": "audio", "muted": false, "locked": false }
  ],
  "clips": [],
  "subtitles": []
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `tracks[].kind` | string | `"video"` 或 `"audio"` |
| `tracks[].muted` | boolean | 轨道是否静音 |
| `tracks[].locked` | boolean | 轨道是否锁定（锁定后不可编辑） |

---

### TimelineClip（片段）— 核心数据结构

```json
{
  "id": "clip-{timestamp}-{random9}",
  "assetId": "asset-xxx",
  "type": "video",
  "trackIndex": 0,
  "startTime": 0,
  "duration": 5.0,
  "trimStart": 0,
  "trimEnd": 0,
  "speed": 1.0,
  "reversed": false,
  "muted": false,
  "volume": 100,
  "opacity": 100,
  "flipH": false,
  "flipV": false,
  "linkedClipIds": [],
  "transitionIn": { "type": "none", "duration": 0 },
  "transitionOut": { "type": "none", "duration": 0 },
  "colorCorrection": {
    "brightness": 0, "contrast": 0, "saturation": 0,
    "temperature": 0, "tint": 0, "exposure": 0,
    "highlights": 0, "shadows": 0
  },
  "effects": [],
  "asset": null,
  "textStyle": null
}
```

**时间与播放控制：**

| 字段 | 类型 | 范围 | 说明 |
|---|---|---|---|
| `type` | string | — | `"video"` / `"image"` / `"audio"` / `"adjustment"` / `"text"` |
| `trackIndex` | int | 0-4 | 轨道索引。**V1=0, V2=1, V3=2, A1=3, A2=4** |
| `startTime` | float | ≥0 | 在时间轴上的起始时间（秒） |
| `duration` | float | >0 | 播放时长（秒） |
| `trimStart` | float | ≥0 | 从素材头部裁掉的秒数 |
| `trimEnd` | float | ≥0 | 从素材尾部裁掉的秒数 |
| `speed` | float | >0 | 播放速度。`0.25`=四分之一速, `0.5`=慢放, `1.0`=正常, `2.0`=快进, `4.0`=四倍速 |
| `reversed` | boolean | — | `true` = 倒放 |

**音频控制：**

| 字段 | 类型 | 范围 | 说明 |
|---|---|---|---|
| `muted` | boolean | — | `true` = 静音 |
| `volume` | int | 0-100 | 音量百分比 |

**视觉控制：**

| 字段 | 类型 | 范围 | 说明 |
|---|---|---|---|
| `opacity` | int | 0-100 | 透明度。100=不透明, 0=完全透明 |
| `flipH` | boolean | — | 水平翻转 |
| `flipV` | boolean | — | 垂直翻转 |

**关联：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `assetId` | string | 引用的素材 ID |
| `linkedClipIds` | string[] | 关联的片段 ID 列表（音视频联动，移动一个另一个跟着） |
| `asset` | object? | 可设为 `null`，前端会根据 `assetId` 自动关联 |

---

### Transition（转场）

`transitionIn` 和 `transitionOut` 字段的结构：

```json
{ "type": "dissolve", "duration": 0.5 }
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | string | 转场类型，见下表 |
| `duration` | float | 转场时长（秒），通常 0.3-1.5 |

**转场类型：**

| 值 | 效果 |
|---|---|
| `"none"` | 无转场（硬切） |
| `"dissolve"` | 溶解/交叉淡化 |
| `"fade-to-black"` | 淡入/淡出黑色 |
| `"fade-to-white"` | 淡入/淡出白色 |
| `"wipe-left"` | 从右向左擦除 |
| `"wipe-right"` | 从左向右擦除 |
| `"wipe-up"` | 从下向上擦除 |
| `"wipe-down"` | 从上向下擦除 |

---

### ColorCorrection（调色）

所有值范围 **-100 到 100**，默认 **0**。

| 字段 | 说明 |
|---|---|
| `brightness` | 亮度。正值提亮，负值压暗 |
| `contrast` | 对比度。正值增强对比，负值降低 |
| `saturation` | 饱和度。正值增强色彩，负值减弱，-100=黑白 |
| `temperature` | 色温。正值偏暖（黄/橙），负值偏冷（蓝） |
| `tint` | 色调偏移。正值偏品红，负值偏绿 |
| `exposure` | 曝光。正值过曝，负值欠曝 |
| `highlights` | 高光。正值提高高光区域亮度，负值压低 |
| `shadows` | 阴影。正值提亮暗部，负值压暗 |

---

### Effects（特效）

`effects` 是一个数组，每个元素：

```json
{
  "type": "blur",
  "intensity": 50,
  "enabled": true,
  "mask": null
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | string | 特效类型，见下表 |
| `intensity` | int | 强度 0-100 |
| `enabled` | boolean | 是否启用 |
| `mask` | object? | 可选遮罩（限定特效作用区域） |

**特效类型：**

| 值 | 效果 |
|---|---|
| `"blur"` | 高斯模糊 |
| `"sharpen"` | 锐化 |
| `"glow"` | 辉光/发光 |
| `"vignette"` | 暗角（四周变暗） |
| `"grain"` | 胶片颗粒噪点 |
| `"lut-cinematic"` | LUT 预设：电影感 |
| `"lut-vintage"` | LUT 预设：复古 |
| `"lut-bw"` | LUT 预设：黑白 |
| `"lut-cool"` | LUT 预设：冷色调 |
| `"lut-warm"` | LUT 预设：暖色调 |
| `"lut-muted"` | LUT 预设：低饱和 |
| `"lut-vivid"` | LUT 预设：高饱和鲜艳 |

**遮罩（Mask）结构：**

```json
{
  "shape": "rectangle",
  "x": 50, "y": 50,
  "width": 30, "height": 30,
  "rotation": 0,
  "feather": 10,
  "inverted": false
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `shape` | string | `"rectangle"` 或 `"ellipse"` |
| `x`, `y` | float | 中心点位置（百分比 0-100） |
| `width`, `height` | float | 尺寸（百分比 0-100） |
| `rotation` | float | 旋转角度（度） |
| `feather` | float | 边缘羽化程度 0-100 |
| `inverted` | boolean | `true` = 反转遮罩（内部不受影响，外部受影响） |

---

### SubtitleClip（字幕）

```json
{
  "id": "sub-{timestamp}-{random9}",
  "text": "字幕内容",
  "startTime": 1.0,
  "endTime": 4.0,
  "trackIndex": 0,
  "style": {
    "fontSize": 32,
    "fontFamily": "sans-serif",
    "fontWeight": "normal",
    "color": "#FFFFFF",
    "backgroundColor": "transparent",
    "position": "bottom",
    "italic": false
  }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | string | 字幕文字内容 |
| `startTime` | float | 显示起始时间（秒） |
| `endTime` | float | 显示结束时间（秒） |
| `trackIndex` | int | 字幕轨道索引 |
| `style.fontSize` | int | 字号（像素） |
| `style.fontFamily` | string | 字体，如 `"sans-serif"`, `"Inter"`, `"Arial"` |
| `style.fontWeight` | string | 字重。`"normal"` / `"bold"` |
| `style.color` | string | 文字颜色（十六进制如 `"#FFFFFF"`） |
| `style.backgroundColor` | string | 背景色。`"transparent"` = 无背景 |
| `style.position` | string | 显示位置。`"top"` / `"center"` / `"bottom"` |
| `style.italic` | boolean | 是否斜体 |

---

### TextStyle（文字叠加）

用于 `type: "text"` 的片段，放在 `clip.textStyle` 字段中：

```json
{
  "text": "标题文字",
  "fontSize": 64,
  "fontFamily": "Inter, Arial, sans-serif",
  "fontWeight": "bold",
  "color": "#FFFFFF",
  "backgroundColor": "transparent",
  "positionX": 50,
  "positionY": 50,
  "opacity": 100,
  "strokeColor": "#000000",
  "strokeWidth": 2,
  "shadowColor": "rgba(0,0,0,0.5)",
  "shadowBlur": 4,
  "letterSpacing": 0,
  "lineHeight": 1.2
}
```

| 字段 | 类型 | 范围 | 说明 |
|---|---|---|---|
| `text` | string | — | 显示的文字内容 |
| `fontSize` | int | >0 | 字号（像素） |
| `fontFamily` | string | — | 字体族 |
| `fontWeight` | string | — | `"normal"` / `"bold"` / `"100"`-`"900"` |
| `color` | string | — | 文字颜色（十六进制） |
| `backgroundColor` | string | — | 背景色 |
| `positionX` | float | 0-100 | 水平位置（百分比，50=居中） |
| `positionY` | float | 0-100 | 垂直位置（百分比，50=居中） |
| `opacity` | int | 0-100 | 透明度 |
| `strokeColor` | string | — | 描边颜色 |
| `strokeWidth` | float | ≥0 | 描边宽度 |
| `shadowColor` | string | — | 阴影颜色 |
| `shadowBlur` | float | ≥0 | 阴影模糊半径 |
| `letterSpacing` | float | — | 字间距（像素） |
| `lineHeight` | float | >0 | 行高倍数 |

---

## 常用操作配方

### 1. 生成视频并添加到时间轴

```
步骤:
1. generate_video(prompt="一只猫在沙滩上奔跑") → {"video_path": "/path/to/video.mp4"}
2. get_project(project_id) → 拿到当前 project JSON
3. 构造 Asset 对象，插入 project.assets 数组头部
4. 构造 TimelineClip 对象，设置 assetId 指向刚才的 Asset
5. 计算 startTime（= 时间轴上已有片段的末尾时间）
6. 将 clip 插入 project.timelines[activeTimeline].clips
7. update_project(project_id, 修改后的 JSON)
```

### 2. 音视频对齐

```
让视频 clip 和音频 clip 的 startTime 设为相同值，
双方的 linkedClipIds 中互相引用对方的 id。
```

### 3. 顺序排列多个片段

```
clip1.startTime = 0
clip2.startTime = clip1.startTime + clip1.duration
clip3.startTime = clip2.startTime + clip2.duration
```

### 4. 生成 ID 的规则

```
timestamp = Date.now() 毫秒时间戳
random9 = Math.random().toString(36).substr(2, 9) 的随机字符串

Project:  "project-{timestamp}-{random9}"
Asset:    "asset-{timestamp}-{random9}"
Timeline: "timeline-{timestamp}-{random9}"
Clip:     "clip-{timestamp}-{random9}"
Subtitle: "sub-{timestamp}-{random9}"
```

---

## 注意事项

1. **先读后改**：修改项目前务必先 `get_project` 读取最新状态
2. **ID 不可变**：不要修改已有对象的 `id` 字段
3. **asset 字段**：clip 中的 `asset` 字段设为 `null` 即可，前端根据 `assetId` 自动关联
4. **文件路径**：所有路径必须是绝对路径，URL 使用 `file://` 协议
5. **GPU 要求**：视频/图片生成需要 NVIDIA GPU（≥32GB VRAM）+ 已下载模型
6. **FFmpeg 要求**：视频导出依赖系统 FFmpeg
7. **单次生成**：后端同一时刻只能运行一个生成任务，需等上一个完成或取消后才能开始下一个
