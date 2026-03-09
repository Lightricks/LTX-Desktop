# Desktop + Palette Integration Design v2 (Final)

**Date:** 2026-03-06
**Status:** Final -- incorporates all answers from the Palette team questionnaire
**Replaces:** `2026-03-06-palette-desktop-integration-design.md`

---

## 1. Goal

Director's Desktop is the local creative powerhouse (GPU rendering, NLE editor, unlimited
local gallery). Director's Palette is the cloud creative hub (storyboards, library,
characters, credits, web/mobile access). This design connects them so a user signed
in to Palette can browse their cloud library from Desktop, push local work to the cloud,
pull cloud assets locally, and use Palette's prompt expander -- all while Desktop remains
fully functional offline.

---

## 2. System Context

```
Director's Palette (Next.js 15, Vercel)
  Production: directors-palette-v2.vercel.app
  Auth: Supabase cookie sessions (web)
        API keys for Desktop (dp_xxx, SHA-256 hashed, admin-only today)
  DB:   Supabase PostgreSQL (gallery, storyboard_characters,
        style_guides, reference, user_credits, brands)
  Storage: Supabase Storage bucket "directors-palette" (public)
  Limits: 500 image cap, videos expire 7 days, 50 MB upload max
  No CORS headers today (must be added for /api/desktop/* routes)
  No staging environment
        |
        |  HTTPS  /api/desktop/*  (new routes, to be built on Palette side)
        |         Authorization: Bearer dp_xxx
        v
Director's Desktop (Electron + FastAPI)
  Electron main process
    - Registers directorsdesktop:// protocol handler
    - Stores token in safeStorage
    - Passes token to backend via POST /api/sync/connect on startup
  FastAPI backend (localhost:8000)
    - Proxies all Palette API calls (frontend never calls Palette directly)
    - Token held in memory: AppState.app_settings.palette_api_key
    - Persisted to settings.json (encrypted at rest via Electron safeStorage)
  React frontend
    - Calls localhost:8000 only
    - Cloud features appear conditionally when connected
```

**Key principle:** Desktop works fully offline. Cloud features are additive.

---

## 3. Authentication

### 3.1 Three Auth Methods

```
Method 1: Browser Login (default, best UX)
  Desktop click "Sign In"
    -> Electron opens system browser to:
       directors-palette-v2.vercel.app/auth/desktop-login
    -> User logs in via Supabase (email/password, Google OAuth)
    -> Palette redirects to: directorsdesktop://auth/callback?token=dp_xxx
    -> Electron intercepts protocol, extracts token
    -> Electron calls POST localhost:8000/api/sync/connect {token: "dp_xxx"}
    -> Backend validates via Palette /api/desktop/auth/validate
    -> Settings saved to disk

Method 2: API Key Paste (power users)
  User generates API key in Palette admin panel
    -> Copies dp_xxx key
    -> Pastes into Desktop Settings > Palette Connection
    -> Frontend calls POST /api/sync/connect {token: "dp_xxx"}
    -> Same validation flow as above

Method 3: QR Pairing (future -- mobile)
  Desktop generates short-lived pairing code + QR
    -> Displays QR in modal
    -> Desktop polls Palette: GET /api/desktop/pair/poll?code=XXXX
    -> User scans QR from Palette mobile (already logged in)
    -> Palette associates code with user, returns token on next poll
    -> Desktop receives token, connects
```

### 3.2 Token Format and Validation

- Format: `dp_` prefix + 40 hex chars (e.g., `dp_a1b2c3...`)
- Storage on Palette: SHA-256 hash in `api_keys` table
- No `/api/me` endpoint exists; user info comes from `auth.users.user_metadata`
  (display_name, avatar_url) returned by the validate endpoint

### 3.3 Token Lifecycle

```
Electron startup
  -> Read encrypted token from safeStorage
  -> If token exists, POST /api/sync/connect to backend
  -> Backend validates with Palette (caches user info in SyncHandler._cached_user)
  -> If validation fails (expired/revoked), set connected=false, clear cache
  -> Frontend polls GET /api/sync/status to show auth state

Sign out
  -> POST /api/sync/disconnect
  -> Backend clears palette_api_key from AppState
  -> Settings saved (key removed from disk)
  -> Electron clears safeStorage
```

### 3.4 What Palette Must Build

| Item | Description |
|------|-------------|
| `/api/desktop/auth/validate` | POST, accepts `Authorization: Bearer dp_xxx`, returns user_metadata |
| `/auth/desktop-login` page | Login flow that redirects to `directorsdesktop://auth/callback?token=dp_xxx` |
| Non-admin API key generation | Allow regular users to create API keys (currently admin-only) |
| CORS headers on `/api/desktop/*` | Not strictly needed (Desktop backend proxies), but good practice |

---

## 4. Desktop Backend Routes

All cloud features are proxied through the Desktop backend. The frontend never calls
Palette directly. This keeps auth tokens server-side and allows offline graceful degradation.

### 4.1 Existing Routes (enhance)

| Route | Method | Current | Enhancement |
|-------|--------|---------|-------------|
| `GET /api/sync/status` | GET | Returns `{connected, user}` | Add `credits_balance`, `gallery_count`, `video_expiry_warning` |
| `POST /api/sync/connect` | POST | Stores token, validates | No changes needed |
| `POST /api/sync/disconnect` | POST | Clears token | No changes needed |
| `GET /api/sync/credits` | GET | Returns `{connected, balance}` | Add `pricing` object with per-model costs |

### 4.2 New Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `GET /api/sync/gallery` | GET | Proxy paginated gallery list from Palette |
| `POST /api/sync/gallery/upload` | POST | Proxy multipart upload to Palette (50 MB max) |
| `GET /api/sync/gallery/{id}/download` | GET | Download a cloud asset to local outputs dir |
| `GET /api/sync/library/characters` | GET | Proxy character list (flattened from per-storyboard) |
| `GET /api/sync/library/styles` | GET | Proxy style guides + brands |
| `GET /api/sync/library/references` | GET | Proxy references (people/places/props/layouts) |
| `POST /api/sync/prompt/enhance` | POST | Proxy to Palette prompt expander |

### 4.3 Route Details

#### GET /api/sync/gallery

Proxies to Palette `GET /api/desktop/gallery?page=1&per_page=50&type=all`.

Response:
```json
{
  "items": [
    {
      "id": "uuid",
      "filename": "gen_001.png",
      "url": "https://...supabase.co/storage/v1/object/public/directors-palette/...",
      "type": "image",
      "size_bytes": 1234567,
      "created_at": "2026-03-06T...",
      "expires_at": null,
      "is_video": false
    }
  ],
  "total": 142,
  "page": 1,
  "per_page": 50,
  "total_pages": 3
}
```

Notes:
- Videos have `expires_at` set (7-day expiry from Palette).
- No thumbnails exist on Palette; Desktop must show full images or generate local thumbnails.
- 500 image cap is enforced server-side on Palette; Desktop shows remaining quota in gallery header.

#### POST /api/sync/gallery/upload

Accepts multipart form data with a file field. Proxies to Palette `POST /api/upload-file`.
50 MB max enforced on both Desktop (FastAPI request size limit) and Palette.

Request: `multipart/form-data` with `file` field.

Response:
```json
{
  "status": "ok",
  "gallery_id": "uuid",
  "url": "https://...supabase.co/storage/..."
}
```

Error when at cap:
```json
{
  "error": "Gallery full (500/500). Delete items in Palette to make room."
}
```

#### GET /api/sync/gallery/{id}/download

Downloads the cloud asset and saves it to Desktop's local outputs directory.
Returns the local file path so the frontend can display it immediately.

Response:
```json
{
  "status": "ok",
  "local_path": "/path/to/outputs/cloud_abc123.png",
  "type": "image"
}
```

#### GET /api/sync/library/characters

Proxies to Palette `GET /api/desktop/library/characters`.

Palette stores characters per-storyboard (`storyboard_characters` table). The Desktop
API endpoint flattens these into a single list, deduplicating by name. Each character
has a single reference image via a gallery FK.

Response:
```json
{
  "characters": [
    {
      "id": "uuid",
      "name": "Maya",
      "role": "protagonist",
      "description": "A young filmmaker...",
      "reference_image_url": "https://...supabase.co/storage/...",
      "storyboard_name": "Episode 1",
      "source": "cloud"
    }
  ]
}
```

#### GET /api/sync/library/styles

Proxies to Palette `GET /api/desktop/library/styles`.

Returns style guides (global to user, table `style_guides`) and brands (table `brands`).
Also includes the 9 hardcoded presets from Palette.

Response:
```json
{
  "styles": [...],
  "brands": [...],
  "presets": ["cinematic", "anime", "noir", ...]
}
```

#### GET /api/sync/library/references

Proxies to Palette `GET /api/desktop/library/references`.

Categories: people, places, props, layouts. Tags searchable via GIN index on Palette.

Query params: `?category=people&search=sunset`

Response:
```json
{
  "references": [
    {
      "id": "uuid",
      "name": "Beach sunset",
      "category": "places",
      "tags": ["sunset", "ocean", "warm"],
      "image_url": "https://...supabase.co/storage/...",
      "source": "cloud"
    }
  ]
}
```

#### POST /api/sync/prompt/enhance

Proxies to Palette `POST /api/prompt-expander`.

When Desktop is connected to Palette, prompt enhancement uses Palette's GPT-4o-mini
expander (richer, director-style output). When disconnected, falls back to the existing
local Gemini-based `EnhancePromptHandler`.

Request:
```json
{
  "prompt": "a woman walking in rain",
  "level": "2x",
  "director_style": "spielberg"
}
```

Palette expander supports:
- `level`: "2x" (moderate) or "3x" (maximum expansion)
- `director_style`: optional style influence (e.g., "spielberg", "kubrick", "nolan")

Response:
```json
{
  "enhanced_prompt": "A determined young woman in a dark trenchcoat...",
  "source": "palette"
}
```

Fallback (disconnected):
```json
{
  "enhanced_prompt": "...",
  "source": "gemini"
}
```

### 4.4 Credits Enhancement

`GET /api/sync/credits` enhanced response:

```json
{
  "connected": true,
  "balance_cents": 4250,
  "balance_display": "$42.50",
  "pricing": {
    "image": 20,
    "video": 40
  },
  "unit": "cents"
}
```

Credits are only deducted for cloud-based generation (Replicate API calls through Palette).
Local GPU generation is always free.

---

## 5. Data Flow Diagrams

### 5.1 Auth Flow (Browser Login)

```
User                 Desktop Frontend    Desktop Backend    Electron Main    Palette Web
  |                       |                    |                 |               |
  |--click "Sign In"----->|                    |                 |               |
  |                       |--IPC: openPaletteLoginPage---------->|               |
  |                       |                    |                 |--open browser->|
  |                       |                    |                 |               |
  |                       |                    |       (user logs in via Supabase)
  |                       |                    |                 |               |
  |                       |                    |                 |<--redirect-----|
  |                       |                    |                 | directorsdesktop://
  |                       |                    |                 | auth/callback?token=dp_xxx
  |                       |                    |                 |               |
  |                       |                    |<--POST /api/sync/connect--------|
  |                       |                    |   {token: "dp_xxx"}             |
  |                       |                    |                 |               |
  |                       |                    |----validate-----|-------------->|
  |                       |                    |    GET /api/desktop/auth/validate
  |                       |                    |<---user_metadata-|--------------|
  |                       |                    |                 |               |
  |                       |                    |--save settings->|               |
  |                       |                    |                 |--safeStorage  |
  |                       |                    |                 |               |
  |                       |<--{connected:true}--|                 |               |
  |<--show user avatar----|                    |                 |               |
```

### 5.2 Cloud Gallery Browse + Download

```
User                 Desktop Frontend    Desktop Backend         Palette API
  |                       |                    |                      |
  |--click Cloud tab----->|                    |                      |
  |                       |--GET /api/sync/gallery?page=1----------->|
  |                       |                    |--GET /api/desktop/gallery------>|
  |                       |                    |<--{items, total, ...}-----------|
  |                       |<--gallery items----|                      |
  |<--render grid---------|                    |                      |
  |                       |                    |                      |
  |--click download------>|                    |                      |
  |                       |--GET /api/sync/gallery/{id}/download---->|
  |                       |                    |--download file from Supabase--->|
  |                       |                    |<--binary file data-------------|
  |                       |                    |--save to outputs/   |
  |                       |<--{local_path}-----|                      |
  |<--show in Local tab---|                    |                      |
```

### 5.3 Push Local Asset to Cloud

```
User                 Desktop Frontend    Desktop Backend         Palette API
  |                       |                    |                      |
  |--click "Push to Cloud" on local asset----->|                      |
  |                       |--POST /api/sync/gallery/upload----------->|
  |                       |   (multipart: file from local path)       |
  |                       |                    |--POST /api/upload-file--------->|
  |                       |                    |   (multipart, 50MB max)         |
  |                       |                    |<--{gallery_id, url}-------------|
  |                       |<--{status: "ok"}----|                      |
  |<--show cloud badge----|                    |                      |
```

### 5.4 Prompt Enhancement (Dual Path)

```
User                 Desktop Frontend    Desktop Backend         Palette API
  |                       |                    |                      |
  |--click sparkle btn--->|                    |                      |
  |                       |--POST /api/sync/prompt/enhance----------->|
  |                       |   {prompt, level, director_style}         |
  |                       |                    |                      |
  |                       |          [connected to Palette?]          |
  |                       |                    |                      |
  |                       |              YES:  |--POST /api/prompt-expander---->|
  |                       |                    |   (GPT-4o-mini, director style)|
  |                       |                    |<--{enhanced}-------------------|
  |                       |                    |                      |
  |                       |              NO:   |--Gemini API (local key)        |
  |                       |                    |   (existing EnhancePromptHandler)
  |                       |                    |<--{enhanced}         |
  |                       |                    |                      |
  |                       |<--{enhanced_prompt, source}---------------|
  |<--replace prompt------|                    |                      |
```

---

## 6. Backend Implementation Details

### 6.1 PaletteSyncClient Protocol Extension

The existing `PaletteSyncClient` protocol (`services/palette_sync_client/palette_sync_client.py`)
must be extended with new methods:

```python
class PaletteSyncClient(Protocol):
    # Existing
    def validate_connection(self, *, api_key: str) -> dict[str, Any]: ...
    def get_credits(self, *, api_key: str) -> dict[str, Any]: ...

    # New
    def list_gallery(self, *, api_key: str, page: int, per_page: int,
                     asset_type: str) -> dict[str, Any]: ...
    def download_asset(self, *, api_key: str, asset_id: str) -> bytes: ...
    def upload_asset(self, *, api_key: str, file_data: bytes,
                     filename: str, content_type: str) -> dict[str, Any]: ...
    def list_characters(self, *, api_key: str) -> dict[str, Any]: ...
    def list_styles(self, *, api_key: str) -> dict[str, Any]: ...
    def list_references(self, *, api_key: str, category: str | None,
                        search: str | None) -> dict[str, Any]: ...
    def enhance_prompt(self, *, api_key: str, prompt: str, level: str,
                       director_style: str | None) -> dict[str, Any]: ...
```

The real implementation (`PaletteSyncClientImpl`) calls Palette's `/api/desktop/*` routes.
A `FakePaletteSyncClient` in `tests/fakes/` provides canned responses for testing.

### 6.2 SyncHandler Extension

The existing `SyncHandler` (`handlers/sync_handler.py`) gains new methods mapping 1:1 to
the new routes. Each method:

1. Reads `api_key` from `self._state.app_settings.palette_api_key`
2. Returns `{"connected": false, ...}` if no key
3. Delegates to `self._client.<method>(...)`
4. Returns the response dict

For `gallery/download`, the handler also writes the downloaded bytes to the local outputs
directory, following the same naming convention as `GalleryHandler`.

For `prompt/enhance`, the handler checks connection status. If connected, calls
`self._client.enhance_prompt(...)`. If disconnected, delegates to the existing
`EnhancePromptHandler.enhance()` method (Gemini path).

### 6.3 New Route File

`_routes/sync.py` already exists. The new gallery/library/prompt routes are added
to this same file, keeping the `/api/sync` prefix.

### 6.4 AppState Changes

No new fields on `AppState`. The `palette_api_key` field on `AppSettings` is sufficient.
Cached user info stays in `SyncHandler._cached_user` (already implemented).

### 6.5 Concurrency

Palette API calls are network I/O. They should NOT hold the shared lock. The existing
SyncHandler methods already follow this pattern (no lock usage). New methods follow
the same approach.

---

## 7. Palette Backend Routes (To Be Built)

These are the Next.js API routes that Palette must implement to support Desktop integration.
All routes require `Authorization: Bearer dp_xxx` header. All routes must add CORS
headers allowing `localhost:8000` origin.

| Route | Method | Description |
|-------|--------|-------------|
| `/api/desktop/auth/validate` | POST | Validate API key, return user_metadata |
| `/api/desktop/gallery` | GET | Paginated gallery list (query: page, per_page, type) |
| `/api/desktop/gallery/download/{id}` | GET | Return signed URL or stream asset bytes |
| `/api/desktop/library/characters` | GET | All characters across storyboards (flattened) |
| `/api/desktop/library/styles` | GET | Style guides + brands + preset names |
| `/api/desktop/library/references` | GET | References with category/tag filtering |
| `/api/desktop/credits` | GET | Balance in cents + pricing table |
| `/api/desktop/pair/create` | POST | Create pairing code (for QR flow, future) |
| `/api/desktop/pair/poll` | GET | Poll for completed pairing (future) |

Existing Palette routes that Desktop proxies directly (no `/api/desktop/` wrapper needed):

| Route | Method | Notes |
|-------|--------|-------|
| `/api/upload-file` | POST | Multipart upload, 50 MB max. Needs CORS + API key auth added. |
| `/api/prompt-expander` | POST | GPT-4o-mini expander. Needs CORS + API key auth added. |

---

## 8. Electron Changes

### 8.1 Protocol Handler Registration

In `electron/main.ts` (or equivalent), register the custom protocol on app startup:

```typescript
app.setAsDefaultProtocolClient('directorsdesktop');

// Handle the protocol URL
app.on('open-url', (event, url) => {
  event.preventDefault();
  handleDeepLink(url);
});

// Windows: protocol URL comes via second-instance
app.on('second-instance', (event, argv) => {
  const deepLink = argv.find(arg => arg.startsWith('directorsdesktop://'));
  if (deepLink) handleDeepLink(deepLink);
  // Focus existing window
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  }
});
```

### 8.2 Deep Link Handler

```typescript
async function handleDeepLink(url: string) {
  const parsed = new URL(url);
  if (parsed.hostname === 'auth' && parsed.pathname === '/callback') {
    const token = parsed.searchParams.get('token');
    if (token) {
      // Store in safeStorage for persistence across restarts
      safeStorage.encryptString(token);  // save to disk

      // Send to backend
      await fetch('http://localhost:8000/api/sync/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token }),
      });

      // Notify renderer to refresh auth state
      mainWindow?.webContents.send('auth-state-changed');
    }
  }
}
```

### 8.3 Startup Token Restoration

On app launch, after the backend is ready:

```typescript
const encryptedToken = readFromDisk();  // safeStorage file
if (encryptedToken) {
  const token = safeStorage.decryptString(encryptedToken);
  await fetch('http://localhost:8000/api/sync/connect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token }),
  });
}
```

### 8.4 New IPC Handler

Add to `electron/preload.ts`:

```typescript
openPaletteLoginPage: () => ipcRenderer.invoke('open-palette-login-page'),
// Already exists in preload.ts -- just needs the main process handler
// to open: directors-palette-v2.vercel.app/auth/desktop-login
```

### 8.5 Electron Builder Config

Add protocol association to `electron-builder.yml`:

```yaml
protocols:
  - name: "Directors Desktop"
    schemes:
      - directorsdesktop
```

---

## 9. Frontend Changes

### 9.1 Gallery View (Gallery.tsx)

Add tabs: **Local** | **Cloud**

- **Local tab** (existing): Shows local gallery from `GET /api/gallery/local`
- **Cloud tab** (new): Shows cloud gallery from `GET /api/sync/gallery`
  - Paginated grid with lazy loading
  - Each item shows a cloud badge icon
  - Click to preview (full-size, no thumbnails from Palette)
  - "Download" button saves to local and switches to Local tab
  - Shows video expiry warning: "Expires in X days" for video items
  - Shows quota: "142 / 500 images"
  - Disabled with "Sign in to access cloud gallery" when not connected

**Push to Cloud** button on local items:
- Available on Local tab items
- Grayed out when not connected or at 500 cap
- Triggers `POST /api/sync/gallery/upload`

### 9.2 Characters View (Characters.tsx)

Show two sections or a toggle: **Local** | **Cloud**

- **Local characters** (existing): From `GET /api/library/characters`
- **Cloud characters** (new): From `GET /api/sync/library/characters`
  - Shows storyboard origin as subtitle: "from Episode 1"
  - Cloud badge on items
  - "Pull to Local" downloads reference image and creates a local character
  - Palette characters are per-storyboard; Desktop flattens into a single list,
    deduplicating by name (keep most recently modified)

### 9.3 Styles View (Styles.tsx)

Similar Local/Cloud split:
- Cloud styles from `GET /api/sync/library/styles`
- Includes brands (separate from style guides on Palette)
- Shows 9 hardcoded presets (always available, no pull needed)
- "Pull to Local" creates a local style entry with downloaded reference image

### 9.4 References View (References.tsx)

- Cloud references from `GET /api/sync/library/references`
- Category filter: people / places / props / layouts (matches Palette categories)
- Tag search (GIN-indexed on Palette, passed as query param)
- "Pull to Local" downloads image, creates local reference

### 9.5 Home.tsx Sidebar

Enhance the account section at the bottom of the sidebar:

```
Connected state:
  [Avatar] display_name
  Credits: $42.50
  [Sign Out]

Disconnected state:
  [Sign In to Director's Palette]
  (or paste API key in Settings)
```

### 9.6 Settings View

Add "Director's Palette" section:

```
Director's Palette
  Status: Connected as "John Doe"    [Disconnect]
  --or--
  Status: Not connected              [Sign In]  [Paste API Key]

  API Key: dp_xxxx...xxxx            [Clear]

  Prompt Enhancement: [Palette (GPT-4o-mini)] / [Local (Gemini)]
    When connected, defaults to Palette. Toggle to override.
```

### 9.7 Prompt Enhancement UI

The sparkle button next to the prompt field in Playground.tsx / GenSpace.tsx:

- Calls `POST /api/sync/prompt/enhance` (unified endpoint)
- Shows source badge: "Enhanced by Palette" or "Enhanced by Gemini"
- Level selector dropdown: "2x" or "3x"
- Optional director style selector (only when Palette connected)

---

## 10. Generation Pipeline: Last Frame Wiring

The `lastFramePath` field exists on `GenerateVideoRequest` but is not yet wired through
the generation pipeline. This section specifies how to complete the wiring.

### 10.1 Queue Pipeline

`QueueSubmitRequest.params` already passes arbitrary params. The `lastFramePath` key
needs to flow through:

```
Frontend: POST /api/queue/submit
  params: { prompt: "...", imagePath: "first.png", lastFramePath: "last.png", ... }
    |
    v
JobQueue.submit() -> Job record with params dict
    |
    v
Job runner picks job, reads params["lastFramePath"]
    |
    v
VideoGenerationHandler or API client
```

### 10.2 LTX Local (Fast Pipeline)

LTX supports multiple `ImageConditioningInput` entries. Currently, first frame is:

```python
ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)
```

Last frame adds a second entry:

```python
ImageConditioningInput(path=last_frame_path, frame_idx=num_frames - 1, strength=1.0)
```

Both can be provided simultaneously for start-to-end guided generation.

The `num_frames` value comes from the duration and fps settings:
`num_frames = int(float(duration) * float(fps)) + 1`

### 10.3 Seedance (Replicate API)

Seedance supports `last_frame` as a separate parameter (not an image conditioning list).
The `ReplicateVideoClientImpl` must accept and pass `last_frame` to the Replicate API
input params.

### 10.4 Receive Job from Palette

The existing `ReceiveJobHandler` already handles `last_frame_url` on `ReceiveJobRequest`.
It downloads the URL to a temp file and passes it as `lastFramePath` in the job params.
This path just needs to be wired through to the generation handler as described above.

---

## 11. Key Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Desktop works fully offline | Cloud is additive, not required |
| 2 | Delete local does not delete cloud | One-way protection, avoids accidental data loss |
| 3 | Push to cloud is explicit | User chooses what uploads; respects 500 cap |
| 4 | Credits only for cloud generation | Local GPU is free; credits = Palette API costs |
| 5 | Videos expire after 7 days on Palette | Desktop warns with countdown badge on video items |
| 6 | Characters are per-storyboard in Palette, global in Desktop | Desktop flattens + deduplicates for simpler browsing |
| 7 | Prompt enhancement: Palette when connected, Gemini when not | Palette uses GPT-4o-mini with director styles; Gemini is the offline fallback |
| 8 | Frontend never calls Palette directly | All proxied through Desktop backend; keeps tokens server-side |
| 9 | Token stored in Electron safeStorage | OS-level encryption; backend holds in memory only |
| 10 | No thumbnails from Palette | Display full images; consider generating local thumbnails for performance |
| 11 | Palette base URL: `directors-palette-v2.vercel.app` | No staging env; single production target |
| 12 | Upload limit: 50 MB | Matches Palette's existing `POST /api/upload-file` limit |

---

## 12. Error Handling

### 12.1 Network Failures

All sync routes return graceful disconnected responses on network failure:

```json
{
  "connected": false,
  "error": "Could not reach Director's Palette. Working offline."
}
```

Frontend shows a subtle toast, not a blocking error. Local features remain fully functional.

### 12.2 Token Expiry/Revocation

If Palette returns 401 on any proxy call:

1. Backend clears `palette_api_key` from AppState
2. Clears `_cached_user`
3. Returns `{"connected": false, "error": "Session expired. Please sign in again."}`
4. Frontend transitions to disconnected state

### 12.3 Gallery Quota

Upload attempts when at 500 cap return:

```json
{
  "error": "gallery_full",
  "message": "Gallery full (500/500). Delete items in Director's Palette to make room.",
  "current": 500,
  "limit": 500
}
```

### 12.4 Video Expiry

When listing cloud gallery items, videos with `expires_at` within 24 hours get flagged:

```json
{
  "expiry_warning": "This video expires in 6 hours. Download locally to keep it."
}
```

---

## 13. Testing Strategy

Following the existing backend testing patterns (integration-first, no mocks):

### 13.1 FakePaletteSyncClient

New file: `tests/fakes/fake_palette_sync_client.py`

Provides canned responses for all `PaletteSyncClient` protocol methods. Supports
configurable scenarios:

- `FakePaletteSyncClient(connected=True)` -- returns valid user, gallery, etc.
- `FakePaletteSyncClient(connected=False)` -- raises on all calls
- `FakePaletteSyncClient(gallery_full=True)` -- upload returns 409
- `FakePaletteSyncClient(token_expired=True)` -- returns 401

### 13.2 Test Cases

**Auth tests:**
- Connect with valid token -> connected=true, user info cached
- Connect with invalid token -> connected=false, error returned
- Disconnect -> clears key, status shows disconnected
- Token expiry on subsequent call -> auto-disconnect

**Gallery tests:**
- List cloud gallery (paginated)
- Download cloud asset to local
- Upload local asset to cloud
- Upload when at 500 cap -> error
- Gallery operations when disconnected -> graceful error

**Library tests:**
- List cloud characters (flattened from per-storyboard)
- List cloud styles + brands + presets
- List cloud references with category filter

**Prompt enhancement tests:**
- Enhance with Palette connected -> uses Palette expander, source="palette"
- Enhance with Palette disconnected -> uses Gemini, source="gemini"
- Enhance with no Gemini key and disconnected -> error

**Credits tests:**
- Get credits when connected -> balance + pricing
- Get credits when disconnected -> graceful error

---

## 14. Migration from v1 Design

This document replaces `2026-03-06-palette-desktop-integration-design.md`. Key changes:

1. **Removed speculative features** not informed by the questionnaire (bidirectional job sync, inpainting, prompt library sync)
2. **Added concrete Palette schema details** (table names, column names, storage bucket, caps)
3. **Added prompt expander integration** as a first-class feature (GPT-4o-mini, 2x/3x levels, director styles)
4. **Clarified character model mismatch** (per-storyboard vs global) with flatten strategy
5. **Added video expiry handling** (7-day expiry with warnings)
6. **Specified all Palette routes to be built** (previously vague)
7. **Added upload details** (multipart, 50 MB, `/api/upload-file` endpoint)
8. **Clarified credits format** (balance in cents, per-model pricing: image 20c, video 40c)
9. **Removed assumptions about existing endpoints** (`/api/me` does not exist; user info from `auth.users.user_metadata`)
10. **Added CORS requirement** for Palette `/api/desktop/*` routes

---

## 15. Implementation Order

1. **Palette side first:** Build `/api/desktop/*` routes + CORS + API key auth for non-admin users
2. **Desktop backend:** Extend `PaletteSyncClient` protocol + impl + sync handler + routes
3. **Electron:** Register protocol handler + deep link + safeStorage flow
4. **Frontend:** Settings connection UI -> Gallery Cloud tab -> Library cloud sections -> Prompt enhance
5. **Last frame wiring:** Queue pipeline -> LTX handler -> Seedance handler
6. **Tests:** FakePaletteSyncClient + integration tests for all sync routes
