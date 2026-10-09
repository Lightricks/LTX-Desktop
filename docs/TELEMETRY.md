# Telemetry

LTX Desktop sends a small amount of anonymous usage data so the team can see how the app is used and where it fails. That is the only purpose. The data is used to improve the app: which features people reach for, which settings they pick, and which generations fail or get cancelled. It is not used for advertising, it is not sold, and it is not linked to an account, a name, or an email address.

No personal information, generated content, prompts, negative prompts, file paths, custom LoRA file names, or IP-derived location is collected.

A random installation ID is created on the first event and stored on the device. It lets us tell that several events came from the same install. It is not an account, and it cannot be used to identify a person.

## What is collected

Every event includes the app version, a device timestamp, the installation ID, and the platform (macOS, Windows, or Linux).

**App launch** (`launched`). Sent once per process for a packaged app. On a first install it is sent when the user clicks Install, after the opt-out on that screen has been saved. On later opens it is sent when the app starts. Also includes total system memory, and the GPU name when it is available.

**Generation start** (`generate_started`). Sent when a generation begins, including one that is cancelled immediately after it is claimed. Also includes total system memory and the GPU name. The operational details are:

- A random generation ID, and the attempt number when a job is retried
- Where it ran: the desktop app or a remote client on the local network, and whether it started from Home or Gen Space
- What was requested: feature (for example text-to-video, image-to-video, audio-to-video, multi-keyframe, retake, extend, or a Home LoRA recipe), image or video, local or LTX API execution, model, resolution, frames per second, and requested duration
- How many images were requested, for an image generation
- LoRA catalog IDs for built-in adapters. A custom LoRA is recorded only as a boolean, never as a file name or path
- How the prompt was passed: `raw` (sent as typed), `manually-enhanced` (the user ran Enhance), or `auto-enhanced` (the app rewrote it before generation). The prompt text is never included

**Generation end** (`generate_ended`). Sent when that same generation succeeds, fails, or is cancelled. It does not repeat the start details. It carries only:

- The same generation ID and attempt, so it can be joined to the start
- The outcome: succeeded, failed, or cancelled
- How long the run took, in milliseconds
- An error code when the run failed, such as a capability or executor failure. The code is a fixed label, not an error message, a prompt, or a path

## Opting out

Analytics is enabled by default. On the initial setup screen, **Opt out of basic analytics** is unchecked. Check it to turn collection off before you click Install. You can also turn it off or back on at any time in **Settings > General > Anonymous Analytics**. When disabled, no events are sent.

To disable telemetry before the first launch, create an `app_state.json` file in the app data folder, or set the flag if that file is already there:

```json
{ "analyticsEnabled": false }
```

App data folder locations:

- **Windows:** `%LOCALAPPDATA%\LTXDesktop\`
- **macOS:** `~/Library/Application Support/LTXDesktop/`
- **Linux:** `$XDG_DATA_HOME/LTXDesktop/` (default: `~/.local/share/LTXDesktop/`)

This method is relevant mainly to machines that are ineligible for local generation. Those machines skip the setup screen and so never show the checkbox. Your preference is respected immediately — no restart required. Development builds do not send events.

## Implementation

Electron is the only process that builds or sends the ingest payload. [`electron/analytics-payload.ts`](../electron/analytics-payload.ts) defines the event names and the detail keys that are allowed onto a start event. [`shared/analytics-ended-details.json`](../shared/analytics-ended-details.json) is the list of keys allowed onto an end event; both the Python forwarder and Electron check that list. [`electron/analytics.ts`](../electron/analytics.ts) applies opt-out, hardware on launch and start events, retries, and the rate limit, then sends the event over HTTPS. No third-party analytics SDKs are used.

The Python backend does not talk to the ingest host. It forwards an event name and operational details to the loopback receiver in [`electron/analytics-sink.ts`](../electron/analytics-sink.ts), which accepts generation start and end events and drops every other detail key. [`electron/main.ts`](../electron/main.ts) sends `launched` once per process: at startup when setup is already complete, when the user clicks Install on a first install, and when an API-only first run finishes setup without that screen. The renderer only signals that click, after the opt-out has been saved. It cannot attach details or choose the ingest payload.
