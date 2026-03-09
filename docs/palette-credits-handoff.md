# Director's Desktop — Credits & Cost Display Integration

**Date:** 2026-03-09
**From:** Desktop team
**To:** Palette engineering team
**Status:** Needed for v1 launch — users need to see costs before generating

---

## The Problem

Desktop users generate videos and images using cloud APIs that cost credits. Right now they have **no visibility into how much anything costs** before they hit Generate, and **no running balance** visible while they work.

We need three things from Palette to fix this:

1. A way to fetch the user's current credit balance
2. A pricing table so we can show "this will cost X credits" before generation
3. A pre-check endpoint so we can block generation when credits are insufficient

---

## What Desktop Needs

### 1. `GET /api/desktop/credits` — Balance + Pricing

We already have this in our spec (`docs/palette-api-spec.md` lines 595-620). Here's the contract:

**Auth:** Bearer token (JWT or `dp_` API key)

**Response `200`:**

```json
{
  "balance_cents": 1500,
  "lifetime_purchased_cents": 5000,
  "lifetime_used_cents": 3500,
  "pricing": {
    "video_t2v": 40,
    "video_i2v": 40,
    "video_seedance": 80,
    "image": 20,
    "image_edit": 20,
    "audio": 15,
    "text_enhance": 3
  }
}
```

**Why we need granular pricing keys:** Desktop supports multiple generation types and models. A flat "video: 40" doesn't work when Seedance costs more than LTX. We need to show the user "This Seedance generation will cost 80 credits" vs "This LTX generation will cost 40 credits."

**Pricing keys we need (at minimum):**

| Key | What it covers | Desktop UI context |
|-----|---------------|-------------------|
| `video_t2v` | Text-to-video (LTX local/API) | User types prompt, hits Generate |
| `video_i2v` | Image-to-video (LTX) | User uploads image + prompt |
| `video_seedance` | Seedance cloud video | User selects Seedance model |
| `image` | Text-to-image (ZIT) | User generates image |
| `image_edit` | Image editing/img2img | User edits existing image |
| `text_enhance` | Prompt enhancement | User clicks magic wand |

If your pricing is simpler (same cost for all video types), just return the same value for all video keys. We'd rather have too many keys than not enough.

**How Desktop uses this:**
- We poll this endpoint when the app starts and after each generation
- We display balance in the header bar: `Credits: $15.00`
- We show cost before generation: `This will cost $0.40 (balance: $15.00)`

---

### 2. `POST /api/desktop/credits/check` — Pre-Generation Check

Already in our spec (lines 624-670). Before every cloud generation, Desktop calls this to verify the user can afford it.

**Request:**

```json
{
  "generation_type": "video_seedance",
  "count": 1
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `generation_type` | string | Yes | One of the pricing keys above |
| `count` | int | No | Number of generations (default 1, for bulk) |

**Response `200` (can afford):**

```json
{
  "can_afford": true,
  "cost_cents": 80,
  "balance_cents": 1500,
  "balance_after_cents": 1420
}
```

**Response `200` (insufficient):**

```json
{
  "can_afford": false,
  "cost_cents": 80,
  "balance_cents": 30,
  "balance_after_cents": -50,
  "top_up_url": "https://directorspalette.com/settings/billing"
}
```

**Why `count`?** Desktop supports bulk generation — a user might queue 5 Seedance videos at once. We need to check if they can afford all 5 before submitting, not fail on video #3.

**Why `top_up_url`?** When the user can't afford it, we want to show a "Top up credits" button that opens their browser to Palette's billing page. If this URL is static, we can hardcode it. If it varies per user, return it here.

---

### 3. `POST /api/desktop/credits/deduct` — After Generation Completes

When a generation finishes successfully, Desktop tells Palette to deduct the credits.

**Request:**

```json
{
  "generation_type": "video_seedance",
  "count": 1,
  "metadata": {
    "model": "seedance-1.5-pro",
    "duration_seconds": 5,
    "resolution": "720p",
    "job_id": "abc123"
  }
}
```

**Response `200`:**

```json
{
  "deducted_cents": 80,
  "balance_cents": 1420
}
```

**Response `402` (insufficient — race condition):**

```json
{
  "error": "insufficient_credits",
  "balance_cents": 30
}
```

**Why deduct after, not before?** Generations can fail or be cancelled. We don't want to charge for failed jobs. The check endpoint gates the UI, the deduct endpoint charges after success.

**Why metadata?** So you have an audit trail of what was generated. Optional fields, include whatever is useful for your analytics.

---

## What Desktop Has Already Built

| Component | Status | Notes |
|-----------|--------|-------|
| `GET /api/sync/credits` backend route | Built | Calls Palette, returns balance to frontend |
| `PaletteSyncClient.get_credits()` | Built | HTTP client method, handles errors gracefully |
| Credit display on Home screen | Built | Shows `Credits: $X.XX` when connected |
| Credit display in Settings | Built | Shows balance in Palette connection section |
| Fake test implementation | Built | Returns `{"balance": 5000, "currency": "credits"}` |

**What we'll build once you deploy these endpoints:**

1. Credit balance in the generation header bar (always visible while working)
2. Cost estimate shown on the Generate button: `Generate ($0.40)`
3. Pre-generation check — "Insufficient credits" dialog with "Top up" button
4. Post-generation deduction
5. Bulk generation cost calculation: `5 videos x $0.80 = $4.00`

---

## Questions We Need Answered

1. **Are the credit endpoints deployed?** The spec has been written since Phase 7 but we don't know if they're live.

2. **Is pricing per-model or flat?** Does a Seedance video cost the same as an LTX video? Does resolution/duration affect price?

3. **Who deducts credits?** Two options:
   - **Option A (we prefer):** Desktop calls `/credits/deduct` after successful generation. Desktop controls when charges happen.
   - **Option B:** Palette's prompt-expander and other proxied endpoints auto-deduct. Desktop only needs to read balance.

   Tell us which model you're using so we build accordingly.

4. **What's the `top_up_url`?** Where do users go to buy more credits? Is it always `https://directorspalette.com/settings/billing` or something else?

5. **Free tier / trial credits?** Do new users get any free credits? How many? We need to know for onboarding messaging.

6. **Local generations cost credits?** When a user generates video on their own GPU (LTX local), does that cost Palette credits? Or only cloud API generations?

---

## How to Test

Once endpoints are deployed:

```bash
# Get balance + pricing
curl -H "Authorization: Bearer dp_..." \
  https://directorspalette.com/api/desktop/credits

# Check if user can afford a Seedance video
curl -X POST -H "Authorization: Bearer dp_..." \
  -H "Content-Type: application/json" \
  -d '{"generation_type": "video_seedance", "count": 1}' \
  https://directorspalette.com/api/desktop/credits/check

# Deduct after successful generation
curl -X POST -H "Authorization: Bearer dp_..." \
  -H "Content-Type: application/json" \
  -d '{"generation_type": "video_seedance", "count": 1}' \
  https://directorspalette.com/api/desktop/credits/deduct
```

Let us know when it's live and we'll wire it up on our side.
