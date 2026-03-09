# Director's Palette API Specification for Desktop Integration

**Version:** 1.0
**Date:** 2026-03-06
**Audience:** Director's Palette engineering team
**Status:** Draft

---

## Table of Contents

1. [Overview](#overview)
2. [Priority Order](#priority-order)
3. [Infrastructure Requirements](#infrastructure-requirements)
4. [Authentication Middleware](#authentication-middleware)
5. [Endpoints: Auth](#endpoints-auth)
6. [Endpoints: Gallery](#endpoints-gallery)
7. [Endpoints: Library](#endpoints-library)
8. [Endpoints: Credits](#endpoints-credits)
9. [Endpoints: Prompts](#endpoints-prompts)
10. [Error Format](#error-format)
11. [Open Questions](#open-questions)

---

## Overview

Director's Desktop is an Electron app for local AI video generation. It needs to integrate with Director's Palette (the Next.js web app) so users can:

- **Authenticate** from the desktop app using their existing Palette account
- **Browse** their cloud gallery, characters, styles, references, and brands
- **Upload** locally generated assets to their cloud gallery
- **Check credits** before running API-backed generations
- **Enhance prompts** using the existing prompt expander

All new endpoints live under the `/api/desktop/*` prefix. This isolates Desktop-specific concerns (CORS, API key auth, pairing) from the existing web app routes.

**Integration topology:**

```
Director's Desktop (Electron)
  |
  | HTTP requests to https://directors-palette-v2.vercel.app/api/desktop/*
  | Auth: Bearer token (JWT or dp_ API key)
  |
Director's Palette (Next.js on Vercel)
  |
  | Supabase (auth, DB, storage)
```

---

## Priority Order

Build in this order. Each phase is independently useful.

| Phase | What | Why |
|-------|------|-----|
| **Phase 1** | Infrastructure (CORS, auth middleware, API key expansion) | Nothing works without this |
| **Phase 2** | `GET /api/desktop/me` | Desktop can verify a token is valid |
| **Phase 3** | Auth pairing flow (pair, poll, complete) + deep link redirect | Users can log in from Desktop |
| **Phase 4** | `GET /api/desktop/gallery` + `DELETE` | Users can browse/manage cloud gallery |
| **Phase 5** | `POST /api/desktop/gallery/upload` | Users can push local generations to cloud |
| **Phase 6** | Library endpoints (characters, styles, references, brands) | Users can pull creative assets into Desktop |
| **Phase 7** | Credits endpoints | Desktop can gate API-backed generations |
| **Phase 8** | `POST /api/desktop/prompt/enhance` | Prompt enhancement from Desktop |

---

## Infrastructure Requirements

### 1. CORS on `/api/desktop/*`

All routes under `/api/desktop/*` must return CORS headers allowing requests from the Desktop backend.

**Required headers on every response (including preflight):**

```
Access-Control-Allow-Origin: http://localhost:8000
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Max-Age: 86400
```

For `OPTIONS` preflight requests, return `204 No Content` with the above headers.

**Implementation note:** In Next.js App Router, this is best done with a middleware function that matches `/api/desktop/:path*` and injects headers. Alternatively, each route handler can call a shared `withCors()` helper. Either pattern is fine as long as every route and every OPTIONS response is covered.

### 2. API Key Expansion

Currently, `api_keys` table entries are admin-only. This must change:

- **Any authenticated user** can create API keys for their own account.
- Existing table schema (`api_keys` with `dp_` + 32-hex format, SHA-256 hash lookup) is fine. No schema changes needed.
- Add two internal endpoints (used by the Palette web UI, not by Desktop):
  - Create key: inserts a row with `user_id`, `hashed_key`, `name`, `created_at`
  - Revoke key: soft-deletes or hard-deletes by key ID for the authenticated user
- Desktop does not call these endpoints directly. The user creates/manages keys in the Palette web UI and pastes the key into Desktop.

### 3. Deep Link Redirect from OAuth Callback

Modify the existing `/auth/callback` route to support a `redirect_to` query parameter:

1. User clicks "Login with Palette" in Desktop.
2. Desktop opens the system browser to: `https://directors-palette-v2.vercel.app/auth/login?redirect_to=directorsdesktop://auth/callback`
3. User completes OAuth (Google/Apple) or email/password login as normal.
4. `/auth/callback` receives the Supabase auth code, exchanges it for a session.
5. If `redirect_to` starts with `directorsdesktop://`, redirect to: `directorsdesktop://auth/callback?access_token={JWT}&refresh_token={JWT}`
6. If `redirect_to` is missing or does not start with `directorsdesktop://`, proceed with existing behavior.

**Security:** Only allow `redirect_to` values starting with `directorsdesktop://`. Reject all other custom schemes.

---

## Authentication Middleware

Every `/api/desktop/*` endpoint requires authentication. The middleware must support two token formats in the `Authorization` header:

### Token Format 1: Supabase JWT

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

- Verify using Supabase's `createClient` / `getUser()` server-side.
- Extract `user.id` from the verified token.

### Token Format 2: API Key

```
Authorization: Bearer dp_a1b2c3d4e5f6...
```

- Detect by the `dp_` prefix.
- SHA-256 hash the key, look up in `api_keys` table.
- Extract `user_id` from the matched row.
- Reject if no match or if the key is revoked/expired.

### Middleware Behavior

```
function authenticateDesktop(request):
    token = request.headers["Authorization"]?.replace("Bearer ", "")

    if not token:
        return 401 { error: "missing_token", message: "Authorization header required" }

    if token.startsWith("dp_"):
        user_id = lookupApiKey(sha256(token))
        if not user_id:
            return 401 { error: "invalid_api_key", message: "API key not found or revoked" }
    else:
        user = supabase.auth.getUser(token)
        if error:
            return 401 { error: "invalid_token", message: "JWT expired or invalid" }
        user_id = user.id

    // Attach user_id to request context for handler use
```

### Authenticated Context

Every handler receives a resolved `user_id: string` (UUID). Handlers never deal with token parsing directly.

---

## Endpoints: Auth

### `GET /api/desktop/me`

Validate the token and return normalized user information.

**Auth:** Required

**Request:** No parameters.

**Response `200`:**

```json
{
  "id": "uuid",
  "email": "user@example.com",
  "display_name": "Jane Smith",
  "avatar_url": "https://lh3.googleusercontent.com/...",
  "created_at": "2025-06-15T10:30:00Z"
}
```

**Field sources:**
- `id`: `auth.users.id`
- `email`: `auth.users.email`
- `display_name`: `auth.users.raw_user_meta_data->>'full_name'` (fall back to email prefix if null)
- `avatar_url`: `auth.users.raw_user_meta_data->>'avatar_url'` (nullable)
- `created_at`: `auth.users.created_at`

**Errors:**
- `401` if token is invalid (see middleware)

---

### `POST /api/desktop/auth/pair`

Create a new pairing session. Desktop displays the resulting QR code. The user scans it with their phone or clicks the URL in their browser (where they are already logged in to Palette).

**Auth:** Not required (this is how unauthenticated Desktop instances initiate login).

**Request body:** None.

**Response `201`:**

```json
{
  "pairing_code": "A1B2C3",
  "qr_url": "https://directors-palette-v2.vercel.app/pair/A1B2C3",
  "expires_at": "2026-03-06T12:15:00Z"
}
```

**Implementation notes:**
- `pairing_code`: 6 alphanumeric characters, uppercase. Unique, not guessable.
- Store in a `pairing_sessions` table (or KV/cache): `code`, `status` (pending/completed/expired), `access_token` (null until completed), `created_at`, `expires_at`.
- Expires after 10 minutes.
- Consider using Supabase table or Vercel KV. Either works.

**Errors:**
- `429` if rate limited (max 5 pairing requests per IP per minute)

---

### `GET /api/desktop/auth/pair/[code]`

Poll the status of a pairing session. Desktop calls this every 2-3 seconds until status is `completed` or `expired`.

**Auth:** Not required.

**Request params:**
- `code` (path parameter): The 6-character pairing code.

**Response `200` (pending):**

```json
{
  "status": "pending",
  "expires_at": "2026-03-06T12:15:00Z"
}
```

**Response `200` (completed):**

```json
{
  "status": "completed",
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "v1.MjQ1..."
}
```

**Response `200` (expired):**

```json
{
  "status": "expired"
}
```

**Implementation notes:**
- Once status is `completed` and the tokens have been returned, immediately delete or invalidate the pairing session. Tokens must only be retrievable once.
- Return `expired` for codes past their `expires_at` or that have already been consumed.

**Errors:**
- `404` if code does not exist: `{ "error": "not_found", "message": "Pairing code not found" }`

---

### `POST /api/desktop/auth/pair/[code]/complete`

Called from the Palette web app (or mobile) by an authenticated user to complete a pairing session. This is what "approves" the Desktop login.

**Auth:** Required (the user completing the pairing must be logged in to Palette).

**Request params:**
- `code` (path parameter): The 6-character pairing code.

**Request body:** None. The user identity comes from the authenticated session.

**Response `200`:**

```json
{
  "status": "completed"
}
```

**Implementation notes:**
- Look up the pairing session by `code`. Verify it is still `pending` and not expired.
- Generate a new Supabase session/token pair for the authenticated user (or reuse the current session tokens).
- Store `access_token` and `refresh_token` in the pairing session row. Set status to `completed`.
- The next poll from Desktop on `GET /api/desktop/auth/pair/[code]` will pick up the tokens.

**Errors:**
- `404` if code does not exist or is expired
- `409` if code is already completed: `{ "error": "already_completed", "message": "This pairing code has already been used" }`

---

## Endpoints: Gallery

### `GET /api/desktop/gallery`

Paginated list of the user's gallery items.

**Auth:** Required

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | integer | `1` | Page number (1-indexed) |
| `per_page` | integer | `50` | Items per page. Max `100`. |
| `type` | string | (all) | Filter by `generation_type`. Values: `image`, `video`, `audio`, or omit for all. |
| `folder_id` | string | (none) | Filter by folder. Omit for all folders. |
| `sort` | string | `created_at_desc` | Sort order. Values: `created_at_desc`, `created_at_asc`. |

**Response `200`:**

```json
{
  "items": [
    {
      "id": "uuid",
      "generation_type": "image",
      "status": "completed",
      "public_url": "https://xyz.supabase.co/storage/v1/object/public/directors-palette/generations/uid/image_abc.png",
      "file_size": 2048576,
      "mime_type": "image/png",
      "metadata": { "prompt": "a sunset over mountains", "model": "flux-schnell" },
      "folder_id": "uuid-or-null",
      "created_at": "2026-03-01T14:22:00Z",
      "updated_at": "2026-03-01T14:22:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 50,
    "total_items": 237,
    "total_pages": 5
  }
}
```

**Notes:**
- Only return items where `user_id` matches the authenticated user.
- Exclude items with `status = 'error'` unless explicitly requested (future filter).
- `public_url` is the Supabase public bucket URL. No signed URLs needed.

**Errors:**
- `400` if `per_page` > 100 or `type` is invalid

---

### `POST /api/desktop/gallery/upload`

Upload a locally generated asset to the user's cloud gallery.

**Auth:** Required

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | The file to upload. Max 50 MB. |
| `generation_type` | string | Yes | `image`, `video`, or `audio` |
| `metadata` | string (JSON) | No | Stringified JSON with prompt, model, parameters, etc. |

**Response `201`:**

```json
{
  "id": "uuid",
  "generation_type": "video",
  "status": "completed",
  "public_url": "https://xyz.supabase.co/storage/v1/object/public/directors-palette/generations/uid/video_abc.mp4",
  "file_size": 15728640,
  "mime_type": "video/mp4",
  "metadata": { "prompt": "a cat playing piano", "model": "ltx-video" },
  "created_at": "2026-03-06T09:00:00Z"
}
```

**Implementation notes:**
- Upload to Supabase storage bucket `directors-palette` at path `generations/{userId}/{type}_{uniqueId}.{ext}`
- Insert a row into the `gallery` table with `status = 'completed'`, `user_id`, `public_url`, `file_size`, `mime_type`, `generation_type`, and `metadata`.
- Enforce the 500-image cap. If the user is at the limit, return `409`.
- For video uploads, set `expires_at` to 7 days from now (matching existing policy).

**Errors:**
- `400` if file is missing or exceeds 50 MB
- `400` if `generation_type` is invalid
- `409` if user has reached the 500-item gallery cap: `{ "error": "gallery_limit_reached", "message": "Gallery limit of 500 items reached. Delete items to upload more." }`
- `413` if file exceeds 50 MB (may be caught at infra level before handler)

---

### `DELETE /api/desktop/gallery/[id]`

Delete a gallery item.

**Auth:** Required

**Request params:**
- `id` (path parameter): Gallery item UUID.

**Response `200`:**

```json
{
  "deleted": true
}
```

**Implementation notes:**
- Verify `gallery.user_id` matches the authenticated user before deleting.
- Delete the file from Supabase storage as well as the database row.
- If the gallery item is referenced by a `storyboard_characters.reference_gallery_id` or `style_guides.reference_gallery_id` or `reference.gallery_id`, either cascade-null those FKs or return an error. Recommendation: cascade-null and delete.

**Errors:**
- `404` if item does not exist or does not belong to user
- `404` (not `403`) for items belonging to other users -- do not reveal existence

---

## Endpoints: Library

### `GET /api/desktop/library/characters`

List all characters across all of the user's storyboards.

**Auth:** Required

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `storyboard_id` | string | (all) | Filter to a specific storyboard |

**Response `200`:**

```json
{
  "characters": [
    {
      "id": "uuid",
      "storyboard_id": "uuid",
      "name": "Detective Harlow",
      "description": "Tall, mid-40s, weathered trench coat, sharp eyes",
      "has_reference": true,
      "reference_image_url": "https://xyz.supabase.co/storage/v1/object/public/directors-palette/generations/uid/image_ref.png",
      "mentions": 12,
      "metadata": {}
    }
  ]
}
```

**Implementation notes:**
- Join `storyboard_characters` through their parent storyboards to filter by `user_id`.
- For `reference_image_url`: if `reference_gallery_id` is not null, join to `gallery` and return `public_url`. Otherwise null.
- Sort by `mentions` descending (most-used characters first).

**Errors:**
- `400` if `storyboard_id` is provided but does not belong to the user

---

### `GET /api/desktop/library/styles`

List user's custom style guides plus the 9 preset styles.

**Auth:** Required

**Response `200`:**

```json
{
  "styles": [
    {
      "id": "uuid",
      "name": "Noir Cinematic",
      "description": "High contrast black and white...",
      "style_prompt": "noir cinematic style, high contrast, deep shadows...",
      "reference_image_url": "https://...",
      "is_preset": false,
      "metadata": {}
    },
    {
      "id": "preset_photorealistic",
      "name": "Photorealistic",
      "description": "...",
      "style_prompt": "photorealistic, highly detailed...",
      "reference_image_url": null,
      "is_preset": true,
      "metadata": {}
    }
  ]
}
```

**Implementation notes:**
- Query `style_guides` where `user_id` matches.
- Append the 9 hardcoded preset styles with `is_preset: true` and synthetic IDs prefixed `preset_`.
- For `reference_image_url`: join to `gallery` via `reference_gallery_id` if present.

---

### `GET /api/desktop/library/references`

List reference images with optional category filter.

**Auth:** Required

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `category` | string | (all) | Filter by category: `people`, `places`, `props`, `layouts` |

**Response `200`:**

```json
{
  "references": [
    {
      "id": "uuid",
      "category": "people",
      "tags": ["protagonist", "female", "young"],
      "gallery_item": {
        "id": "uuid",
        "public_url": "https://...",
        "mime_type": "image/png",
        "file_size": 1024000
      }
    }
  ]
}
```

**Implementation notes:**
- Join `reference` to `gallery` via `gallery_id` FK.
- Filter to items where the gallery item's `user_id` matches the authenticated user.
- If `category` is provided, filter by `reference.category`.
- Sort by `reference.id` descending (newest first).

**Errors:**
- `400` if `category` is not one of the four valid values

---

### `GET /api/desktop/library/brands`

List user's brands.

**Auth:** Required

**Response `200`:**

```json
{
  "brands": [
    {
      "id": "uuid",
      "name": "Acme Corp",
      "slug": "acme-corp",
      "logo_url": "https://...",
      "tagline": "Innovation for everyone",
      "industry": "Technology",
      "audience": { "age_range": "25-45", "interests": ["tech", "productivity"] },
      "voice": { "tone": "professional", "personality": "authoritative" },
      "visual_identity": { "primary_color": "#2563EB", "font": "Inter" },
      "visual_style": { "mood": "clean", "photography": "lifestyle" },
      "music": { "genre": "electronic", "tempo": "moderate" },
      "brand_guide_image_url": "https://..."
    }
  ]
}
```

**Implementation notes:**
- Query `brands` table. Filter mechanism depends on how brands are associated with users. If there is a `user_id` column, filter by it. If brands are shared/org-level, return all brands the user has access to.
- Map `audience_json` to `audience`, `voice_json` to `voice`, etc. (drop the `_json` suffix in the API response for cleaner naming).

---

## Endpoints: Credits

### `GET /api/desktop/credits`

Return the user's credit balance and model pricing.

**Auth:** Required

**Response `200`:**

```json
{
  "balance_cents": 1500,
  "lifetime_purchased_cents": 5000,
  "lifetime_used_cents": 3500,
  "pricing": {
    "image": 20,
    "video": 40,
    "audio": 15,
    "text": 3
  }
}
```

**Implementation notes:**
- `balance_cents`, `lifetime_purchased_cents`, `lifetime_used_cents` from `user_credits` table and aggregated from `credit_transactions`.
- `pricing` from `model_pricing` table. Map to the four generation types. Values are in cents.
- This is a superset of the existing `GET /api/credits` response, repackaged for clarity.

---

### `POST /api/desktop/credits/check`

Check whether the user can afford a specific generation type. Does not deduct.

**Auth:** Required

**Request body:**

```json
{
  "generation_type": "video"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `generation_type` | string | Yes | `image`, `video`, `audio`, or `text` |

**Response `200`:**

```json
{
  "can_afford": true,
  "cost_cents": 40,
  "balance_cents": 1500,
  "balance_after_cents": 1460
}
```

**Response `200` (insufficient balance):**

```json
{
  "can_afford": false,
  "cost_cents": 40,
  "balance_cents": 30,
  "balance_after_cents": -10
}
```

**Notes:**
- This is read-only. It does not deduct credits.
- Desktop uses this before starting a generation to show the user a confirmation or a "top up credits" prompt.

**Errors:**
- `400` if `generation_type` is invalid

---

## Endpoints: Prompts

### `POST /api/desktop/prompt/enhance`

Proxy to the existing prompt expander.

**Auth:** Required

**Request body:**

```json
{
  "prompt": "a cat sitting on a windowsill",
  "level": "2x",
  "style": "cinematic"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | The original prompt to enhance. Max 2000 characters. |
| `level` | string | No | Enhancement level: `2x` (default) or `3x` |
| `style` | string | No | Director style to apply (e.g., `cinematic`, `anime`, `documentary`). Omit for general enhancement. |

**Response `200`:**

```json
{
  "original_prompt": "a cat sitting on a windowsill",
  "enhanced_prompt": "A fluffy orange tabby cat perched gracefully on a sun-drenched windowsill, warm golden hour light streaming through sheer curtains, dust motes floating in the air, shallow depth of field, intimate documentary photography style",
  "level": "2x",
  "style": "cinematic"
}
```

**Implementation notes:**
- This is a thin proxy over the existing `POST /api/prompt-expander` logic. Extract the shared logic into a function callable from both routes, or have this route call the existing one internally.
- Deduct text-generation credits (3 cents) per call.

**Errors:**
- `400` if `prompt` is empty or exceeds 2000 characters
- `400` if `level` is not `2x` or `3x`
- `402` if user has insufficient credits: `{ "error": "insufficient_credits", "message": "Not enough credits for prompt enhancement", "cost_cents": 3, "balance_cents": 1 }`

---

## Error Format

All errors follow a consistent shape:

```json
{
  "error": "error_code_snake_case",
  "message": "Human-readable description of what went wrong"
}
```

### Standard Error Codes

| HTTP Status | Error Code | When |
|-------------|-----------|------|
| `400` | `bad_request` | Malformed input, invalid parameters |
| `401` | `missing_token` | No Authorization header |
| `401` | `invalid_token` | JWT expired or invalid |
| `401` | `invalid_api_key` | API key not found or revoked |
| `402` | `insufficient_credits` | User cannot afford the operation |
| `404` | `not_found` | Resource does not exist (or user lacks access) |
| `409` | `already_completed` | Pairing code already used |
| `409` | `gallery_limit_reached` | 500-item gallery cap hit |
| `413` | `file_too_large` | Upload exceeds 50 MB |
| `429` | `rate_limited` | Too many requests |
| `500` | `internal_error` | Unexpected server error |

---

## Open Questions

These are decisions for the Palette team. Desktop will adapt to whatever you choose.

1. **Pairing session storage:** Supabase table vs. Vercel KV? KV is simpler for ephemeral data with TTL. A Supabase table is fine too.

2. **Brands ownership:** The `brands` table schema provided does not include a `user_id` column. How are brands associated with users? Is there a join table? Desktop needs to know which brands to return.

3. **API key management UI:** Desktop assumes users create/revoke API keys in the Palette web settings page. Does the Palette team want to add this to an existing settings view, or create a new one?

4. **Token refresh:** When Desktop receives a JWT via pairing or deep link, the JWT will eventually expire. Should Desktop call a `POST /api/desktop/auth/refresh` endpoint with the refresh token, or should it just re-trigger the pairing/login flow? Recommendation: add a refresh endpoint, but it is not blocking for Phase 1.

5. **Prompt enhancement styles:** What are the valid `style` values for the prompt expander? Desktop needs an enum or a list endpoint. For now, Desktop will send freeform strings and let the expander handle unknowns gracefully.

6. **Video expiry enforcement:** Desktop-uploaded videos get `expires_at` set to 7 days. Is there an existing cron/scheduled function that cleans up expired items, or does Desktop need to handle this awareness client-side only?
