# API keys, cost, and privacy

All API keys are optional. Local generation does not require one. The app stores keys locally in your app data folder, so treat them like secrets. When you use an API-backed feature, the prompts and media for that feature are sent to the API service.

## LTX API key

The LTX API is used for:

- **Cloud text encoding** — **FREE**. It is highly recommended because it speeds up inference and saves memory.
- **Cloud prompt enhancement** — optional. Controlled by **Auto-enhance on API generation** in **Settings > General > Generation**.
- **Cloud video generation** in Gen Space and the Video Editor (LTX 2.5 and 2.3, Fast and Pro) — paid.
- **Retake and Extend in the Video Editor** — paid.

Generate a free API key at the [LTX Console](https://console.ltx.video/). Text encoding is free; video generation API usage is paid. [Read more](https://ltx.io/model/model-blog/ltx-2-better-control-for-real-workflows).

## fal API key (optional)

Used for Z Image Turbo text-to-image generation and image editing in Gen Space. When enabled, image requests are sent to fal.ai. You can opt in from **Settings > Legacy > Image generation**, even on hardware that supports local generation. Create a key in the [fal dashboard](https://fal.ai/dashboard/keys).

## Gemini API key (optional)

Used for the **Enhance** button's **Gemini API** provider (see [Prompt enhancement (local)](tools-and-options.md#prompt-enhancement-local)) and for AI prompt suggestions in the Video Editor timeline gap fill. It also powers the [Assistant](../README.md#assistant) when Gemini is the selected model. When enabled, the prompt and any frames or reference images in the request are sent to Google Gemini.
