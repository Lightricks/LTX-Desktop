# Director's Desktop — Palette Team Handoff

**Date:** 2026-03-08
**From:** Desktop team
**To:** Palette engineering team
**Status:** Blocking — Desktop auth is broken without this

---

## What's happening

Director's Desktop has a Settings screen where users connect their Palette account. Right now, **it doesn't work**. Users enter their `dp_` API key, hit Save, and nothing happens — it silently fails and shows "Not connected."

We've built a workaround (email/password login via Supabase directly), but `dp_` API keys — which is what users actually have — won't work until you deploy one endpoint.

---

## What we need from you (priority order)

### 1. `GET /api/desktop/me` (BLOCKING — do this first)

This is the only endpoint that is currently blocking us. Everything else can wait.

**What it does:** Validates a token and returns user info.

**Auth:** Required — `Authorization: Bearer {token}`

The token will be either:
- A Supabase JWT (from `supabase.auth.getSession()`)
- A `dp_` API key (from the `api_keys` table)

**Your auth middleware needs to handle both formats:**

```
if token starts with "dp_":
    hash = sha256(token)
    lookup hashed_key in api_keys table
    get user_id from matched row
    reject if no match or revoked
else:
    user = supabase.auth.getUser(token)
    reject if invalid/expired
    user_id = user.id
```

**Response `200`:**

```json
{
  "id": "uuid",
  "email": "user@example.com",
  "display_name": "Jane Smith",
  "avatar_url": "https://lh3.googleusercontent.com/..."
}
```

**Field sources:**
- `id`: `auth.users.id`
- `email`: `auth.users.email`
- `display_name`: `auth.users.raw_user_meta_data->>'full_name'` (fall back to email prefix)
- `avatar_url`: `auth.users.raw_user_meta_data->>'avatar_url'` (nullable)

**Error responses:**
- `401` if token missing: `{ "error": "missing_token" }`
- `401` if token invalid: `{ "error": "invalid_token" }`
- `401` if API key not found: `{ "error": "invalid_api_key" }`

---

### 2. CORS on `/api/desktop/*`

Desktop's Python backend (running on `localhost:8000`) makes HTTP requests to Palette. You need CORS headers on every `/api/desktop/*` response, including OPTIONS preflight:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Max-Age: 86400
```

We're using `*` for origin because the request comes from a local Python server, not a browser. If you want to lock it down, `http://localhost:8000` works, but `*` is simpler and fine for server-to-server.

For OPTIONS requests, return `204 No Content` with the above headers.

**In Next.js App Router**, the simplest approach is a middleware that matches `/api/desktop/:path*`.

---

### 3. API key creation UI (nice-to-have, not blocking)

Users need a way to create `dp_` API keys from the Palette web UI. If this doesn't exist yet, users can still use email/password login from Desktop (we built that). But eventually they'll want API keys for a set-it-and-forget-it connection.

A simple settings page section:
- "Create API Key" button → generates `dp_` + 32 hex chars
- Show the key once, store SHA-256 hash in `api_keys` table
- List existing keys with revoke option

---

## What Desktop has already built

Here's the full integration infrastructure on our side, ready to go:

| Feature | Status | What it does |
|---------|--------|-------------|
| Email/password login | Working now | Calls Supabase `/auth/v1/token` directly, stores JWT + refresh token |
| Token auto-refresh | Working now | Refreshes expired JWTs using stored refresh token |
| `dp_` API key auth | Built, waiting on you | Calls `GET /api/desktop/me` — currently gets 404 |
| Settings UI | Working now | Tabbed auth: "Login with Email" + "API Key" tabs, error feedback, disconnect |
| Gallery sync client | Built, waiting on you | Calls `GET /api/desktop/gallery`, `DELETE /api/desktop/gallery/{id}` |
| Library sync client | Built, waiting on you | Calls characters, styles, references endpoints |
| Credits check | Built, waiting on you | Calls `GET /api/desktop/credits` |
| Prompt enhancement | Built, waiting on you | Calls `POST /api/desktop/prompt/enhance` |

**The only thing blocking a user-facing release is #1 (`GET /api/desktop/me`) and #2 (CORS).** Everything else can be phased in.

---

## How Desktop calls you

All requests go from Desktop's Python backend (NOT from the browser/Electron renderer):

```
User clicks "Connect" in Desktop
  → Electron frontend calls localhost:8000/api/sync/connect
  → Python backend calls https://directorspal.com/api/desktop/me
  → With header: Authorization: Bearer dp_a1b2c3d4...
  → Palette validates, returns user info
  → Desktop stores the key and shows "Connected as user@example.com"
```

**Base URL we're hitting:** `https://directorspal.com`

If this isn't right (staging URL, different domain, etc.), let us know and we'll update.

---

## The full API spec

The complete endpoint spec is already written: **`docs/palette-api-spec.md`** in this repo. It covers all 8 phases (auth, gallery, library, credits, prompts) with exact request/response shapes, error codes, and implementation notes.

But **you don't need to read all of that right now.** Just build `/api/desktop/me` + CORS and we're unblocked.

---

## How to test it

Once you deploy `/api/desktop/me`:

```bash
# Test with a Supabase JWT
curl -H "Authorization: Bearer eyJhbG..." \
  https://directorspal.com/api/desktop/me

# Test with a dp_ API key
curl -H "Authorization: Bearer dp_a1b2c3d4e5f6..." \
  https://directorspal.com/api/desktop/me

# Both should return:
# { "id": "...", "email": "...", "display_name": "...", "avatar_url": "..." }
```

Then tell us it's live and we'll test from Desktop.

---

## Questions?

If anything is unclear or you need to change the contract (different URL, different response shape, etc.), reach out before building. We can adapt on our side quickly.
