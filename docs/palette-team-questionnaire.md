# Director's Palette — Integration Questionnaire

**Context:** We're building integration between Director's Desktop (Electron app for local AI video/image generation) and Director's Palette (web/mobile). Desktop needs a set of API endpoints from Palette to enable auth, gallery sync, library sync, and credits. This questionnaire helps us write the exact API spec for your team.

Please answer each section. If something doesn't exist yet, just say "not built yet" — that's useful info too.

---

## 1. Authentication & Users

1a. **What Supabase Auth providers are enabled?** (check all that apply)
- [ ] Email/password
- [ ] Google OAuth
- [ ] Apple OAuth
- [ ] Other: ___________

1b. **Is there a `profiles` or `users` table** beyond Supabase's built-in `auth.users`? If yes, what columns does it have? (We need: display name, email, avatar URL at minimum)

1c. **Do you have an API keys system?** Can users generate long-lived API keys (like "Developer API Keys" in settings)? If yes:
- What table stores them?
- What columns? (key hash, user_id, label, created_at, etc.)
- How are they validated? (lookup by hash? prefix matching?)

1d. **Is there any existing endpoint that validates a token and returns user info?** (e.g., `GET /api/me` or similar) If yes, what's the route and response shape?

1e. **What does a Supabase access token look like in your system?** When a user logs in via browser, do you use:
- Supabase JWT (from `supabase.auth.getSession()`)
- A custom session token
- Something else

---

## 2. Gallery / Generated Assets

2a. **What Supabase Storage bucket(s) store generated images/videos?** (bucket name(s))

2b. **Is there a database table that indexes gallery items?** If yes:
- Table name?
- Key columns? (id, user_id, filename, file_path/storage_key, type, prompt, model_name, created_at, etc.)
- Any RLS policies? (user can only see their own items?)

2c. **What's the current gallery cap?** (The design doc mentions 500 images — is that enforced? Where?)

2d. **Are there thumbnails?** Auto-generated, or same file served at different sizes?

2e. **How are storage URLs generated?**
- Public bucket with direct URLs?
- Private bucket with signed URLs? (if so, what expiry?)
- Through an API proxy?

2f. **What's the max file size for uploads?** Any format restrictions?

---

## 3. Library — Characters

3a. **Is there a `characters` table?** If yes:
- Table name and key columns?
- How are reference images stored? (array of storage paths? separate join table?)
- Is there RLS? (user sees only their characters?)

3b. **What fields does a character have?** (name, role, description, reference_images, etc.)

3c. **Are characters tied to a specific project/brand, or global to the user?**

---

## 4. Library — Styles & Brands

4a. **Is there a `styles` or `brands` table?** If yes:
- Table name and key columns?
- What fields? (name, description, reference_image, color palette, fonts, etc.)

4b. **Are "style guides" a thing in Palette?** (the 3x3 grid generation feature) If yes, where are they stored?

4c. **Is there a brand identity system?** (logos, color palettes, font selections tied to a brand)

---

## 5. Library — References

5a. **Is there a `references` or `reference_images` table?** If yes:
- Table name and key columns?
- Categories? (people, places, props, other — or different?)

5b. **Are references shared across projects or per-project?**

---

## 6. Prompts

6a. **Is there a saved prompts / prompt library feature?** If yes:
- Table name and key columns?
- Fields? (text, tags, category, use_count, etc.)

6b. **Is there a prompt enhancement/expansion feature?** (rough prompt -> detailed cinematic prompt) If yes, what model/API does it use?

---

## 7. Credits

7a. **How does the credit system work?**
- Table/column that stores balance? (e.g., `profiles.credits_balance` or separate `credits` table?)
- Are credits purchased? Subscription-based? Free tier?
- What costs credits? (image generation, video generation, prompt enhancement?)

7b. **Is there an endpoint to check credit balance?** Route and response shape?

7c. **Is there an endpoint to deduct credits?** Or do credits deduct automatically when a generation job runs?

7d. **Credit costs per action:**
- Image generation: ___ credits
- Video generation: ___ credits
- Prompt enhancement: ___ credits
- Other: ___________

---

## 8. Existing API Routes

8a. **List any existing API routes that might be relevant** to this integration. For each, provide:
- Route (method + path)
- What it does
- Auth required? (how?)
- Response shape (or link to code)

Common ones we'd want to know about:
- User profile / me
- Gallery list
- Gallery upload
- Character CRUD
- Style CRUD
- Credits balance
- Prompt library

8b. **What's your API auth pattern?**
- Bearer token in Authorization header?
- Cookie-based sessions?
- Supabase anon key + user JWT?
- Something else?

8c. **Is there CORS configured?** Desktop will call from `http://localhost:8000` (backend proxy) — not from a browser directly.

---

## 9. Technical Details

9a. **What's the production URL for Palette?** (e.g., `https://directorspalette.com`, `https://app.directorspalette.com`, etc.)

9b. **What framework/stack is the API built on?** (Next.js API routes? Separate backend? Supabase Edge Functions?)

9c. **Is there a staging/dev environment** we can test against?

9d. **Any rate limiting on API routes?** If yes, what are the limits?

---

## 10. Desktop-Specific Needs

These are features we want to build. Tell us if they conflict with anything or if you have preferences:

10a. **Browser login redirect:** After login on Palette web, we want to redirect to `directorsdesktop://auth/callback?token=XXX`. Is there an existing OAuth callback flow we should hook into, or do you need to add a redirect?

10b. **QR code pairing:** Desktop shows a QR code, user scans from Palette mobile app. We need a short-lived pairing endpoint. Do you have anything like this, or would it be net-new?

10c. **"Send to Desktop" from Palette:** A button in Palette web that sends a generation job to the user's Desktop app. Any existing job/queue system we should integrate with, or is this net-new?

10d. **Gallery upload from Desktop:** When a user generates locally and clicks "Push to Cloud," we upload the file + metadata. Any preferences on how this should work? (multipart form? presigned URL? base64?)

---

## 11. Anything Else

Anything we should know about the Palette architecture, conventions, or constraints that would affect this integration? Anything planned that might overlap?

---

**Return this completed questionnaire to the Desktop team and we'll produce a detailed API spec for the endpoints we need you to build.**
