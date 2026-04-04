# LTX-Desktop MCP Server

An MCP (Model Context Protocol) server that wraps [LTX-Desktop](https://github.com/Lightricks/LTX-Desktop)'s AI video generation backend and project editing capabilities, enabling AI assistants to automate video creation and timeline editing.

## Features

| Capability | Description |
|---|---|
| 🎬 Video Generation | Text-to-video, image-to-video, audio-to-video |
| 🖼️ Image Generation | Text-to-image with configurable resolution |
| 🔄 Video Retake | AI re-generation of specific video segments |
| 🎞️ Timeline Editing | Full timeline manipulation via project JSON |
| ✂️ Clip Control | Trim, speed, reverse, opacity, volume |
| 🎨 Color Grading | Brightness, contrast, saturation, temperature, exposure |
| ✨ Effects | Blur, sharpen, glow, vignette, grain, LUT presets |
| 🔤 Text & Subtitles | Text overlays and subtitle tracks |
| 🔀 Transitions | Dissolve, fade, wipe (8 types) |
| 📦 Export | FFmpeg-based video export (H.264, ProRes, VP9) |

## Architecture

```
┌─────────────────┐     stdio      ┌──────────────────┐
│   AI Assistant   │◄─────────────►│  MCP Server      │
│ (Claude, etc.)   │               │  ltx_mcp_server.py│
└─────────────────┘               └────┬────────┬─────┘
                                       │        │
                              HTTP :8000│        │HTTP :8100
                                       ▼        ▼
                                 ┌──────────┐ ┌──────────────┐
                                 │ Python   │ │ Electron     │
                                 │ Backend  │ │ Project      │
                                 │ (FastAPI)│ │ Bridge       │
                                 └──────────┘ └──────────────┘
                                  Video Gen    Timeline JSON
                                  Model Mgmt   Export/Import
```

## Prerequisites

- [LTX-Desktop](https://github.com/Lightricks/LTX-Desktop) installed and running
- Python 3.10+
- NVIDIA GPU with ≥32GB VRAM (for local generation)

## Installation

```bash
# 1. Install Python dependencies
pip install mcp httpx

# 2. Install LTX-Desktop dependencies
cd /path/to/LTX-Desktop
pnpm install

# 3. Start LTX-Desktop
pnpm dev
```

## Usage

### Configure in your AI assistant

Add to your MCP configuration (e.g. Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ltx-desktop": {
      "command": "python3",
      "args": ["/path/to/LTX-Desktop/ltx_mcp_server.py"],
      "env": {
        "LTX_BACKEND_URL": "http://127.0.0.1:8000",
        "LTX_BRIDGE_URL": "http://127.0.0.1:8100"
      }
    }
  }
}
```

### Available MCP Tools

#### Video Generation (Backend API)

| Tool | Description |
|---|---|
| `generate_video` | Generate video from text/image/audio |
| `generate_image` | Generate images from text |
| `retake_video` | Re-generate a portion of a video |
| `suggest_gap_prompt` | AI-suggest prompt for timeline gaps |
| `get_generation_progress` | Check generation progress |
| `cancel_generation` | Cancel running generation |

#### Model Management

| Tool | Description |
|---|---|
| `get_health` | Backend health & GPU status |
| `get_models_status` | Model download status |
| `download_models` | Trigger model downloads |

#### Project & Timeline Editing (Electron Bridge)

| Tool | Description |
|---|---|
| `list_projects` | List all projects |
| `get_project` | Get full project JSON |
| `update_project` | Update project (timeline, clips, etc.) |
| `export_video` | Export timeline as video file |
| `import_asset` | Import local file into project |

### Example Workflow

```
User: "Generate 3 video clips about a sunset and arrange them on the timeline with dissolve transitions"

AI Assistant:
1. generate_video(prompt="golden sunset over ocean, waves crashing") → clip1.mp4
2. generate_video(prompt="sun dipping below horizon, orange sky") → clip2.mp4  
3. generate_video(prompt="twilight sky with first stars appearing") → clip3.mp4
4. get_project(project_id) → read current project
5. Add 3 assets + 3 clips to timeline with:
   - clip1: startTime=0, transitionOut={type: "dissolve", duration: 0.5}
   - clip2: startTime=5, transitionIn/Out={type: "dissolve", duration: 0.5}
   - clip3: startTime=10, transitionIn={type: "dissolve", duration: 0.5}
6. update_project(project_id, modified_json)
7. export_video(project_id, "/home/user/sunset_montage.mp4")
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LTX_BACKEND_URL` | `http://127.0.0.1:8000` | LTX-Desktop FastAPI backend URL |
| `LTX_BRIDGE_URL` | `http://127.0.0.1:8100` | Electron project bridge URL |
| `LTX_TIMEOUT` | `600` | Request timeout in seconds |

## Project Structure (MCP additions)

```
LTX-Desktop/
├── ltx_mcp_server.py          # MCP Server (standalone script)
├── electron/
│   ├── project-bridge.ts      # HTTP bridge for project JSON access
│   └── main.ts                # Modified: registers project bridge
└── .agents/skills/
    └── ltx-desktop-mcp/
        └── SKILL.md            # AI assistant skill documentation
```

## License

This MCP integration follows the same license as [LTX-Desktop](https://github.com/Lightricks/LTX-Desktop).
