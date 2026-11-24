# Will it run on my machine?

A plain-language guide to what LTX Desktop can generate on your hardware:
**text/image/audio generation**, **Retake & Extend** (video editing), and
**IC-LoRA** (video recipes and Ingredients). The tables are what the app
itself offers in Cautious mode: they are worked out from the same memory
budget the app checks when you press Generate, and that budget is fitted to
real runs on one RTX 5090 PC and one Mac (an M5 Max). The app refuses jobs it
cannot run rather than failing halfway. On other cards and Macs, and for clips
made in parts, the memory a job needs is calculated from those runs, not run on
that card (see [Brave mode](#brave-mode) for what to do beyond them).

The numbers behind these tables live in the app's code. If what the app does
and what this page says disagree, the app is right and this page is out of
date — please report it.

This guide covers local generation on **Windows PCs with an NVIDIA GPU** and on
**Apple Silicon Macs**. It has one part per OS, each covering generation,
editing, and IC-LoRA.

**On Windows with an NVIDIA GPU?** Read
[On Windows](#on-windows-nvidia-gpu). **On a Mac?** Skip to
[On a Mac](#on-a-mac-apple-silicon).

## Three ways a video can be generated

A video is a 3D box of "tokens": width × height × time. Every 32 × 32 pixel
patch of a frame is one token, and every 8 frames add one more layer in time.
More resolution, a longer clip, or a higher frame rate makes the box bigger,
and a bigger box needs more memory. When the box doesn't fit, the app has two
independent levers:

1. **How the model is loaded** — all of it on the GPU, or streamed in parts
   (the **Full** and **Stream** marks in the tables below).
2. **How the box is sliced** — along time (**chunking**) or across the frame
   (**tiling**), so only a smaller piece sits in memory at once.

### Way 1: Single-pass (Full or Stream)

In a normal single-pass run, the model looks at the whole box at every step.

```text
SINGLE-PASS (normal generation)
Every token attends to every other token across the whole video at once.

      +---------------------------------------+
     /                                       /|
    /                 TIME                  / |  Height
   /              (all frames)             /  |
  +---------------------------------------+   +
  |                                       |  /
  |       100% of video in memory         | /  Width
  |                                       |/
  +---------------------------------------+

  - Memory:  Highest (holds the whole video at once)
  - Speed:   Fastest (no repeated frames, no overlapping pieces)
  - Quality: The reference (the first and last frame see each other directly)
```

There are two loading variants:

- **Full:** all model weights stay on the GPU next to the video. Fastest, most
  memory.
- **Stream:** weights are loaded layer by layer during each step. About
  10–25% slower. **The output pixels are identical to Full** — it only changes
  how the model is moved, not what is generated.

### Way 2: Temporal chunking (slicing time into overlapping windows)

Instead of one giant pass, the app generates the clip in shorter windows, one
after another (for example 129, 97, 73, 49 or 41 frames at a time). If each
window were generated on its own there would be a hard cut between them, so
each window starts from the tail of the one before it — the **carry**
(typically 17 frames for text/image/audio generation, 25 for IC-LoRA).

```text
TEMPORAL CHUNKING (e.g. 97-frame windows with a 25-frame carry)

Timeline (frames):   1                  73    97                169   193       241
                     |-------------------|=====|------------------|=====|---------|
Window 1 (97f):      [====== Gen 1 ======|carry]
                                         :     :
Window 2 (97f):                          [carry|====== Gen 2 =====|carry]
                                                                  :     :
Window 3 (73f):                                                   [carry|== Gen 3 ]

                     [====] = overlap ("carry"): already generated in the previous
                              window and fed into the next one so motion matches.
```

What it costs and buys:

- **Memory drops.** Peak memory follows the window length (97 frames instead
  of 241), plus a small overhead for stitching the windows together. Smaller
  windows need less memory but mean more of them; for text/image/audio
  generation the app starts from the longest window and steps down
  (129 → 97 → 73 → 49 → 41 frames) until one fits.
- **Time goes up.** Every carry region is generated twice: with a 41-frame
  window and a 17-frame carry each window adds only 24 new frames, so a
  20-second (481-frame) clip takes 20 windows. Each window also repeats its own
  upscale-and-decode cycle.
- **Plain text/image-to-video can drift.** Window 5 only sees the tail of
  window 4 and has forgotten window 1, so lighting, background, or a subject's
  look can slowly change over many windows, like a game of telephone.
- **With IC-LoRA it is nearly lossless** (see below): the guide video anchors
  every window.

### Way 3: Spatial tiling (slicing the frame into overlapping tiles)

At very large frame sizes even one window has too many pixels. Tiling cuts
each frame into overlapping rectangles and runs the model on one tile at a
time, inside every denoising step. Neighbouring tiles overlap so their edges
blend without visible seams.

```text
SPATIAL TILING (inside one 4K frame)

  +-------------------+-------+-------------------+
  |                   |       |                   |
  |      Tile 1       |overlap|      Tile 2       |
  |                   |   :   |                   |
  +-------------------+-------+-------------------+
  |      overlap      |   +   |      overlap      |
  +-------------------+-------+-------------------+
  |                   |   :   |                   |
  |      Tile 3       |overlap|      Tile 4       |
  |                   |       |                   |
  +-------------------+-------+-------------------+

  At EVERY denoising step:
    1. Run the model on Tile 1
    2. Run the model on Tile 2
    3. Run the model on Tile 3
    4. Run the model on Tile 4
    5. Blend the overlapping edges together
    --> Go to the next denoising step
```

- **Memory drops during denoising**, because only one tile is in memory at a
  time. The final decode of the whole frame still needs room for the full
  canvas, so the saving is not total.
- **Time goes way up.** A 4K frame in 1920×1088 tiles takes up to 9 tiles per
  step (about 50 seconds per step on an RTX 5090); finer 1024×576 tiles need
  36 per step, roughly four times longer.
- **It needs a guide.** The top-left tile can't see the bottom-right one except
  slowly through the overlap, so for plain text-to-video each tile could invent
  its own scene. With **IC-LoRA** the source video already fixes the layout of
  the whole frame, so tiling works well. That is why tiling is only used for
  IC-LoRA.

### Side by side

```text
+--------------------+------------------+------------------+----------------------------+
| Strategy           | Peak memory      | Generation time  | Output vs. single-pass     |
+--------------------+------------------+------------------+----------------------------+
| 1. Full            | Highest          | Fastest (1.0x)   | Identical (the reference)  |
| 2. Stream          | Medium (1 pass)  | ~1.1x - 1.25x    | Identical pixels           |
| 3. Chunked (large  | Low              | ~1.2x - 1.5x     | Slight drift over a long   |
|    windows)        |                  |                  | video                      |
| 4. Chunked (small  | Very low         | ~2.0x - 3.5x     | More drift (many windows)  |
|    windows)        |                  |                  |                            |
| 5. Tiled (+chunk)  | Lowest per step  | ~3x - 15x slower | Needs a guide (IC-LoRA)    |
+--------------------+------------------+------------------+----------------------------+
```

The dividing line: **Full → Stream** is a transparent loading choice (same
algorithm, same video). **Single-pass → chunking or tiling** changes the
algorithm itself — a different video, and more time.

### Does chunking drift with IC-LoRA?

Much less than in plain text/image-to-video, because every window gets
three anchors:

```text
IC-LORA CHUNKING: THREE ANCHORS PER WINDOW

Window 1 (frames 1..97)               Window 2 (frames 73..169)
+-------------------------------+     +--------------------------------+
| 1. Guide slice (frames 1..97) |     | 1. Guide slice (frames 73..169)| <-- locks motion & layout
| 2. Carry overlap (25 frames)  | --> | 2. Carry overlap (frames 73..97)| <-- smooths the seam
| 3. Reference still (+1 frame) | --> | 3. Reference still (+1 frame)  | <-- locks the overall look
+-------------------------------+     +--------------------------------+
```

- **Geometry and motion don't drift.** Each window receives its own slice of
  your source video, so camera motion, subject position and pose stay locked
  to it for the whole clip.
- **Look and colour depend on the recipe.**
  - _Restoration recipes_ (Restore, Refine Details, Upscale): the source already
    holds the true colours and lighting, so there is virtually no drift.
    Restore deliberately works in 97-frame windows on longer clips, because
    running it unchunked past that length loses detail.
  - _Style and look-transfer recipes_ (Day to Night, Layout to Render, Canny,
    Depth, Pose): the model has to invent colours or textures, so the first
    frame (or your look image) is attached to every window as a reference still
    to keep them consistent.
  - _Outpainting_: the centre is locked by the source, but the invented borders
    rely on the overlap between windows, so subtle texture drift can still
    show in the margins of very long clips.

So chunking is a genuine quality-and-speed trade-off for text/image-to-video,
and nearly free in quality for IC-LoRA — you only pay the time for the overlap.

### Which way the app picks

The app uses the cheapest way that fits your machine: **Full**, then
**Stream**, then **chunking** with the longest window that fits. Tiling is used
only for IC-LoRA (Restore, Refine Details, and the 1440p/4K cells). In
[Brave mode](#brave-mode) the app still steps down through the windows
first; Brave only comes into play when even the smallest window doesn't
fit.

Where each way shows up below:

- **Text/image/audio generation up to 1080p:** single-pass (Full or Stream)
  whenever your machine can hold the clip in one go. When it cannot — a long
  clip, a high frame rate, or a smaller card or Mac — the app chunks it
  instead of refusing, at any resolution.
- **Text/image/audio generation at 1440p and 4K:** chunked, always streamed.
- **IC-LoRA:** single-pass for small jobs, chunked past a size limit; Restore
  and Refine Details are tiled, and the 1440p/4K cells are tiled and chunked.
- **Retake and Extend:** single-pass only — they never chunk or tile, which is
  why long or large edits shorten or refuse instead.

## On Windows (NVIDIA GPU)

### Windows in 30 seconds

- **Text/image/audio generation runs on 10 GB cards and up.** When a clip
  is too big for one pass, the app makes it in overlapping parts instead of
  refusing it: a 12 GB card runs 720p at 20 seconds and 1080p at 10
  seconds in parts, and a 16 GB card runs 1080p at 20 seconds in parts. Those
  numbers are at 24 fps — higher frame rates trim the top (see below). 8 GB
  cards sit below the 9 GB floor — the app routes them to the API
  automatically.
- **1440p and 4K text/image/audio generation runs in overlapping time
  windows and needs a bigger card.** 1440p: up to 10 seconds on 16 GB, 20 on
  24 GB and 32 GB. 4K: up to 10 seconds on 24 GB and 32 GB cards (on 24 GB
  at 24/25 fps; fewer seconds at 48/50 fps). Both are slow — on an RTX 5090 a
  3-second 4K clip takes about 7 minutes and a 10-second one about 26. Jobs
  beyond the lengths we have run need [Brave mode](#brave-mode).
- **Higher frame rates cost more memory.** 25 fps matches 24 fps almost
  everywhere; 48/50 fps render twice the frames, so the longest clips need
  more parts or a bigger card — 1080p at 20 seconds and 48/50 fps needs 24 GB.
- **Ultrawide costs more memory.** Every local clip offers 21:9 through 9:16
  (except 4K, which has no 21:9); the tables show 16:9. Narrower shapes fit wherever 16:9 fits, but 21:9
  ultrawide needs ~30% more and can refuse where 16:9 runs.
- **Video editing (Retake & Extend) needs a 12 GB graphics card or better
  for most jobs.** 10 GB cards stream the smallest cells; 8 GB cards cannot
  run it at all.
- **Bigger picture + longer clip = more parts, and a slower run.** Up to 1080p
  the app picks one pass when it fits and parts when it doesn't, so a smaller
  card trades time, not options. 1080p at 20 seconds runs in 3 parts on 24 GB
  and 32 GB cards and in 6 on a 16 GB card.
- **IC-LoRA recipes take a source clip of up to 10 seconds.** At 540p and
  720p a 16 GB card runs the whole 10 seconds; at 1080p most recipes need
  24 GB for the whole clip (Restore and Refine Details run it on 16 GB).
  Smaller cards get shorter clips (see below).
- **System RAM matters too.** 64 GB runs everything at full speed. On a fast
  NVMe drive, 32 GB runs every tool (including IC-LoRA) and 24 GB runs
  generation and editing (see below).
- **Disk space is an install question, not a per-job one.** The model pack
  needs ~72 GB free to download (see the README); generating itself only
  needs room for the output video, so the app never refuses a job over disk.
- **"Streamed" is not an error.** It means the app fits a bigger job by loading
  the model in parts. It runs a bit slower; that's normal.

### How to read the Windows tables

**Columns — your graphics card, by its memory (VRAM).** Find the column closest
to your card without going over it. Examples: RTX 3080 (10 GB), RTX 4070
(12 GB), RTX 4080 (16 GB), RTX 4090 (24 GB), RTX 5090 (32 GB). 8 GB cards
(RTX 4060 Ti) sit below the 9 GB local-generation floor, so every table starts
at 10 GB.

**Rows — the job you want.** Resolution × length, at 24 fps in 16:9 — the
anchor the generation tables are quoted at. Lengths between two rows fall
between them; the app computes your exact clip when you press Generate. The
frame-rate and aspect-ratio notes under each generation table say how the
other picker values differ.

**Cells — what the app will do:**

| Mark       | Meaning                                                                  |
| ---------- | ------------------------------------------------------------------------ |
| **Full**   | Runs with the whole model on the GPU. Fastest.                           |
| **Stream** | Runs by streaming the model's weights layer by layer. A bit slower. Normal for big jobs. |
| **Parts (n)** | Runs in _n_ overlapping time windows (see [Way 2](#way-2-temporal-chunking-slicing-time-into-overlapping-windows)), always streamed. Slower; plain text-to-video can drift a little across parts. |
| **—**      | Not offered. The app tells you why instead of failing halfway.           |

In the app, a **—** cell shows as a greyed-out option with the reason
attached — the picker tells you before Generate, not after. Frame-rate and
duration pickers are linked: picking 50 fps can grey out long durations that
24 fps offers, each with its reason. Video-editing and IC-LoRA options depend
on your source clip, so the same picker can offer more (or less) once you pick
one. Editing and IC-LoRA follow your source media — they have no frame-rate or
aspect-ratio pickers, so only the generation tables carry those notes.

**Two things that move you one column left in real life:**

1. **Free vs total memory.** The picker and the queue judge your card by its total
   memory, so you see everything it can run. Whether there's room right now is
   checked when the job starts: Generate checks at the click, and a queued job
   checks when its turn comes. Between queued jobs the app first clears what the
   previous job left behind. A browser with 50 tabs can still eat 1–2 GB — if a
   job that "should" fit is refused when it starts, close other apps and try again.
2. **System RAM (below).** Streamed jobs also need enough PC memory; the RAM
   table applies on top of every "Stream" and "Parts" cell.

### Retake & Extend (video editing)

Retake re-generates part of your clip; Extend grows it longer. Both are budgeted
the same way: what counts is the **total footage the AI processes at once**.

The table shows the **longest window of footage** each card processes at once,
at 24 fps, rounded down to the half second. A clip that needs more is shortened to that window instead of being
refused.

| Picture | 10 GB  | 12 GB | 16 GB | 24 GB | 32 GB |
| ------- | ------ | ----- | ----- | ----- | ----- |
| 540p    | 11.5 s | 16 s  | 20 s  | 20 s  | 20 s  |
| 720p    | 6.5 s  | 9 s   | 14.5 s | 20 s | 20 s  |
| 1080p   | 2.5 s  | 3.5 s | 6 s   | 10.5 s | 15.5 s |

20 s is the longest window the editor ever takes. Every cell streams, except on
32 GB cards, which run 540p windows up to 20 s, 720p up to about 10 s and 1080p
up to about 5 s with the whole model on the GPU (**Full**).

**Three things that surprise people:**

- **Extend counts your existing footage too.** A "10 second" extension on top of
  10 seconds of context is budgeted as ~15 seconds of processing. Extensions
  also always keep at least ~5 seconds of your clip as context, so the longest
  single extension is about 16 seconds (less at 1080p on smaller cards).
- **Long clips shorten automatically.** You don't get an error — you get
  the longest window your card can run (the table above).
- **8 GB cards sit out video editing entirely.** The smallest edit needs about
  8.8 GB, and 8 GB cards report under the 9 GB floor. This is a hard floor,
  not a setting we can lower. 10 GB cards stream short windows (the first
  column above) — the app trims to your exact window.

**Above 1080p.** Retake and Extend normally cap the picture at 1080p. With the
experimental **High-resolution video to video** option on, sources up to 4K are
kept (and a 1440p tier appears for a 4K source). At **1440p**, Cautious mode
runs the longest window that fits your card (about **8.5 seconds** of processed
footage on 32 GB, **5.5 seconds** on 24 GB, **3 seconds** on 16 GB, **1.5
seconds** on 12 GB — longer clips shorten automatically, just like at 1080p).
At **4K**, edits never run in windows and Cautious mode runs short windows that
fit your card and the 65-frame video-decode ceiling (about **2.5 seconds** on
32 GB, **2 seconds** on 24 GB, **1 second** on 16 GB, **0.5 seconds** on
12 GB); longer 4K edit selections need [Brave mode](#brave-mode).

### IC-LoRA (video recipes and Ingredients)

Most IC-LoRA tools are **video recipes** (Day to Night, Deblur, Restore,
Refine Details, Layout to Render and the rest): they take your source clip,
up to **10 seconds**, and keep its motion and framing. What counts is the
output size and how much of the clip runs. The table shows the longest
source clip each card runs at 24 fps in Cautious mode; a longer clip is
refused with the reason (Brave mode offers to try it).

| Recipe and output         | 10 GB  | 12 GB | 16 GB  | 24 GB | 32 GB |
| ------------------------- | ------ | ----- | ------ | ----- | ----- |
| Most recipes, 540p        | 3 s    | 5 s   | 8.5 s  | 10 s  | 10 s  |
| Most recipes, 720p        | 1.5 s  | 3 s   | 10 s   | 10 s  | 10 s  |
| Most recipes, 1080p       | —      | 1 s   | 2 s    | 10 s  | 10 s  |
| Layout to Render, 540p    | 3.5 s  | 6 s   | 10 s   | 10 s  | 10 s  |
| Restore, 1080p            | 2 s    | 4 s   | 10 s   | 10 s  | 10 s  |
| Refine Details, 1080p     | 1.5 s  | 5 s   | 10 s   | 10 s  | 10 s  |

Layout to Render matches the other recipes at 720p and 1080p. Restore and
Refine Details run in tiles, and their longer clips run in windows whose
decode is priced on its own, so at 1080p they fit cards the other recipes
don't; at 540p and 720p they run the whole 10 seconds on every card except
Restore on 10 GB cards (2 s).

**Ingredients (from images).** Ingredients builds a clip from your images and
has its own duration picker (5 to 20 seconds). What matters is the images'
shape:

- **Wide image** (landscape, e.g. 1024×576 or any 16:9 image)
- **Tall phone photo** (portrait, e.g. 768×1280)

| Job (image shape × length)   | 10 GB   | 12 GB   | 16 GB  | 24 GB  | 32 GB  |
| ---------------------------- | ------- | ------- | ------ | ------ | ------ |
| Wide image, up to 20 s       | —⁴      | Stream¹ | Stream | Stream | Full   |
| Tall phone photo, up to 5 s  | —       | Stream¹ | Stream | Stream | Full   |
| Tall phone photo, up to 10 s | —       | —       | —²     | Stream | Stream |
| Tall phone photo, 20 s       | —       | —       | —      | —³     | Stream |

¹ Asks ~11.7 GB — tight on a 12 GB card; close other apps first.
² Asks ~16.4 GB — just over a 16 GB card, so it is refused (8 s fits).
³ Asks ~25.9 GB. Needs 32 GB; 24 GB cards are refused, not streamed.
⁴ 10 GB cards run wide images up to 10 s.

**How your image's size enters the budget (Ingredients).** When you upload a photo, the app
reads its pixel dimensions, derives a working grid from them, and checks each
duration against that grid — new upload, new verdicts. Generate checks once
more against the preprocessed clip, so preprocessing can't sneak an
over-the-fence job past the upload check.

On the default path (every catalog recipe as shipped: stage-1-only), the
working grid comes from a fixed 768-pixel-wide bucket — only your photo's
_shape_ sets its height. A 48 MP phone photo and a small web image of the
same shape budget identically, which is why the table covers just two shapes
(wide / tall). Absolute size matters only under Advanced settings: **Source
dimensions** sizes the grid to your photo's own pixels (anything past 1080p
refuses on any card, and 1080p itself asks ~22 GB — so downscale first), while
**two-stage** sizes it to about half your photo's pixels (a 1080p photo fits,
~4K refuses, and the limit is length × photo size together).

One fence applies on every path, any machine: the output must fit inside
1920×1088, turned either way (the experimental 1440p/4K cells below are the
only exception). On the default path that only bites photos
taller than about 3.3:1, and because it is the _shape_ that refuses,
downscaling the file won't help; crop to a less extreme shape instead.

**IC-LoRA at 1440p and 4K (experimental).** Every catalog recipe can render at
1440p or 4K, but only when the **High-resolution video to video** option in the
app's developer panel is on (it ships off). The recipe runs in a single stage,
in tiles of up to 1920×1088 (four tiles per 1440p frame, nine per 4K frame), in
short windows of 41 frames that overlap by 17 — each window adds 24 new frames
(see [Three ways a video can be generated](#three-ways-a-video-can-be-generated)).
The cells never upscale: your source must be at least as large as the cell
(about 2144 px on its short side for 4K, 1424 px for 1440p).

- **What Cautious mode allows at 1440p:** most recipes up to **5 windows** — 137
  frames, about 5 seconds at 24 or 25 fps — on 16 GB cards and up (a 5-second
  clip asks ~13.4 GB). Restore and Refine Details run the whole 10-second clip
  on 24 GB and 32 GB cards; on a 16 GB card Restore stops at 5 seconds and
  Refine Details at 2 seconds. The 16 GB and 24 GB figures are calculated from
  our RTX 5090 runs, not run on those cards.
- **What Cautious mode allows at 4K:** 32 GB cards only. Most recipes and
  Restore run the whole 10-second clip (a 5-second clip asks ~23.3 GB for most
  recipes). Refine Details at 4K has not run cleanly, so Cautious refuses it.
- **Needs Brave mode:** most recipes at 1440p past 5 windows, Refine Details at
  4K, and 1440p clips longer than your card fits. 4K on a card under 32 GB is
  refused in both modes: the decoder cannot finish a 4K frame there.
- **Time and quality:** on an RTX 5090 a 1440p window takes about 13 minutes
  (about 66 minutes for a 5-window clip), and a 4K step takes about 50 seconds
  (~8 minutes per window, over an hour for a 10-second clip). The output has
  1440p or 4K pixels but about 1080p detail (one stage, no refinement pass), and
  a recipe that changes the whole scene (Day to Night, for example) can differ a
  little from tile to tile — check results by eye.

### Text, image & audio generation (T2V/I2V/A2V)

Generate a video from a prompt, a start image, or an audio track. One table
serves all three — the app budgets them identically, and the source image
costs a flat, tiny amount. Shorter clips inside a row may run Full where the
row says Stream, or in one pass where it says Parts; the row shows its longest
length. Every 1440p and 4K cell streams. The table is the 24 fps picture in
16:9 in Cautious mode; the notes below it cover the other frame rates and
shapes.

| Job (resolution × length) | 10 GB   | 12 GB     | 16 GB     | 24 GB      | 32 GB     |
| ------------------------- | ------- | --------- | --------- | ---------- | --------- |
| 270p/360p up to 5 s¹      | Stream² | Stream    | Stream    | Stream     | Full      |
| 540p up to 5 s            | Stream² | Stream    | Stream    | Stream     | Full      |
| 540p up to 10 s           | Stream² | Stream    | Stream    | Stream     | Full      |
| 540p up to 20 s           | —       | Parts (2) | Stream    | Stream     | Stream    |
| 720p up to 5 s            | Stream² | Stream    | Stream    | Stream     | Full      |
| 720p up to 10 s           | —       | Parts (2) | Stream    | Stream     | Full      |
| 720p up to 20 s           | —       | Parts (4) | Parts (2) | Stream     | Stream    |
| 1080p up to 5 s           | —       | Parts (2) | Stream    | Stream     | Stream    |
| 1080p up to 10 s          | —       | Parts (6) | Parts (3) | Stream     | Stream    |
| 1080p up to 20 s          | —       | —⁴        | Parts (6) | Parts (3)  | Parts (3) |
| 1440p up to 10 s          | —       | —         | Parts (7)³ | Parts (3)³ | Parts (2) |
| 1440p up to 20 s          | —       | —         | —⁴        | Parts (6)³ | Parts (5) |
| 4K up to 10 s             | —       | —         | —         | Parts (10)³ | Parts (7) |

¹ The app budgets 270p like 540p and 360p like 720p. Shorter clips can fit
where the row's longest does not: a 10 GB card runs 1080p at 2 s, and a 12 GB
card runs 1440p at 2 s (in 2 parts).
² Exactly at the limit on 10 GB cards — close other apps first. A card that
reports less than its full 10 GB runs none of these. (8 GB cards report under
the 9 GB floor and are routed to the API.)
³ Calculated, not run on that card: we ran 1440p and 4K on an RTX 5090 (32 GB),
so the 32 GB column was run. The memory need on 16 GB and 24 GB cards is
calculated from those runs (the smaller windows those cards use were only tried
on the 32 GB card, forced to stream). Close other apps first and expect long
runs.
⁴ More parts than we have run (12 on a 12 GB card at 1080p, 15 on a 16 GB card
at 1440p): Cautious mode refuses it, and [Brave mode](#brave-mode) runs it with
a warning.

**The headline for 10 GB cards:** text/image generation is fully available up
to 720p/5 s and 540p/10 s — video editing is what wants 12 GB, not this.

**1440p and 4K.** These run in overlapping time windows (see
[Three ways a video can be generated](#three-ways-a-video-can-be-generated)),
always streamed. The app tries the longest window that fits your card:
129 frames on 32 GB at 1440p (and through 8 s on 24 GB), shorter ones on
smaller cards or longer 24 GB clips (97 frames from 10 s at 1440p on 24 GB;
49 frames at 3 s at 4K on 24 GB, stepping down to 41 frames from 4 s), which
means more windows and a slower run. Plain text-to-video
can drift a little between windows. The pickers offer 1440p up to 20 s and 4K
up to 10 s (no 21:9 at 4K; audio-to-video starts at 5 s).

- **What Cautious mode allows.** The app only offers jobs of up to **9
  windows at 1440p** and up to **15 windows at 4K** — the longest runs we have
  run on an RTX 5090. (We also ran 1440p as 20 short 41-frame windows, but the
  app counts windows, not their length, so it stops at 9.) That is what the
  rows above show: 16 GB cards reach 10 s at 1440p (7 windows of 49 frames, 8
  at 25 fps; 4 s runs in 2 windows of 73 frames),
  and 24 GB and 32 GB cards reach 20 s (2 windows of 129 frames at 6 and 8 s
  and 3–6 windows of 97 frames at 10–20 s on 24 GB; 5
  windows of 129 frames on 32 GB); 4K clips of 10 s need 24 GB (2 windows of
  49 frames at 3 s, 4–10 windows of 41 frames at 4–10 s) or 32 GB (7 windows
  of 49 frames). Longer clips, or cards below
  the rows (4K on 16 GB, 1440p past 2 s on 12 GB), are refused in Cautious mode and need
  [Brave mode](#brave-mode).
- **Frame rate.** 25 fps matches 24 fps. At 48/50 fps the clip has twice the
  frames, so more windows: 1440p tops out at 6 s on 16 GB, 10 s on 24 GB and
  20 s on 32 GB; 4K at 6 s on 24 GB, and 10 s on 32 GB.
- **Image-to-video and audio-to-video** are priced the same as text-to-video
  and offered on the same rows (we ran both at 1440p and 4K on the RTX 5090).
- **Time.** On an RTX 5090 a 3-second 4K clip takes about 7 minutes and a
  10-second one about 26, and the app shows a long-run warning for 4K.
  Smaller cards use shorter windows and take longer still.

A clip that fits your card in one pass but is longer than the longest single
pass we ran (past 10 seconds at 1080p and 24 fps, or 540p at 20 seconds and 48/50 fps) runs in two or more
windows in Cautious mode, and in one pass with a warning in Brave mode. After one
clean Brave run on your card, Cautious mode runs that clip in one pass too.

In Brave mode, below 1440p, a clip that can run either way shows a **Make it**
choice: **One pass** or **In parts**. The app picks one, and the meter shows
the memory each needs. One pass has no joins between parts but needs more
memory; parts fit a smaller card. **Never split into parts** (Settings →
Hardware budget) makes every clip in one pass: some lengths are then not
offered, a 1440p or 4K clip longer than one window is refused, and in Brave
mode some clips ask more memory than the card has.

How much of this the app runs without asking depends on the runs behind it.
Up to 9 windows is a tested case on Windows, on any card, and just runs. Past
that, a clip has not been tried at that length: Cautious mode refuses it with a
message saying so, and Brave mode (the default) runs it with a warning that it
may take long, fail or stall. After one clean run on your card, Cautious mode
accepts that clip length there too. If a
card cannot fit even the smallest window (41 frames), the app refuses with the
memory it would need.

**Frame rate (24/25/48/50 fps).** 25 fps matches the table almost
everywhere — the one exception is 540p at 10 seconds, which needs 12 GB at
25 fps (10 GB at 24 fps, exactly at the limit). 48 and 50 fps render twice
the frames, so clips need more parts and the top lengths step down: 10 GB
cards keep 540p up to 5 s (4 s at 50 fps) and 720p up to 3 s; 12 GB cards keep
540p and 720p up to 20 s and 1080p up to 6 s; 1080p at 10 s needs 16 GB and
1080p at 20 s needs 24 GB. On 32 GB cards, 48/50 fps also runs Stream where
the table says Full (540p/10 s, 720p/10 s). The picker lists each frame rate's
durations separately and greys out the ones that don't fit, with the reason
attached.

**Aspect ratio.** Every cell offers 21:9, 16:9, 3:2, 4:3, 1:1, 4:5, and 9:16.
Narrower and square shapes cost the same or less than 16:9, so they fit
wherever the table says the job runs. 21:9 ultrawide is ~30% wider, so it takes
more parts and can refuse where 16:9 runs: 10 GB cards stop at 540p 6 s and
720p 4 s, 12 GB cards at 1080p 8 s, 16 GB cards at 1440p 3 s, and 24 GB cards
at 4K 2 s. The picker greys out a refusing 21:9 shape with its numbers
attached, and queuing one is refused before anything is saved. The model was
trained on 16:9 and 9:16; the picker marks the other ratios with a quality
warning.

### System memory (RAM)

Every "Stream" and "Parts" cell above _also_ needs enough PC memory. (Full loads keep
everything on the graphics card and skip this check.) The app picks the path
automatically from your drive type and memory size:

| Job              | Standard path (any drive, or NVMe with enough RAM) | Lower-RAM NVMe path (NVMe SSD)                                                                           |
| ---------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Text/image/audio | 40 GB minimum                                      | 24 GB minimum (streams from the SSD below 40 GB, ~15–25% slower)                                         |
| Retake & Extend  | 64 GB minimum (non-NVMe drives)                    | 24 GB minimum (NVMe always streams from the SSD; full speed at 64 GB, ~15% slower at 24–32 GB)           |
| IC-LoRA          | 64 GB minimum                                      | 32 GB minimum (streams from the SSD below 64 GB, ~1.3–1.5× slower)                                       |

In practice: **64 GB of RAM runs everything at full speed. On an NVMe
drive, 32 GB runs every tool** (including IC-LoRA, streaming weights from the
SSD), and **24 GB runs generation and editing**. Under 24 GB, streamed jobs are
not offered.

The app enforces these floors.

### What still needs Brave mode (Windows)

Cautious mode refuses these; [Brave mode](#brave-mode) runs them with a
warning:

- Clips that run in more than 9 windows at 1080p and below (for example 1080p
  at 20 s on a 12 GB card), until a clean run on your card.
- 1440p text/image/audio clips past 9 windows (for example 20 s on a 16 GB
  card, or 20 s at 48/50 fps on a 24 GB card) and 4K clips past 15 windows,
  and either on a card below the rows above (4K on 16 GB, 1440p past 2 s on
  12 GB).
- IC-LoRA at 1440p past 5 windows (about 5 seconds; Restore and Refine Details
  run longer on 24 GB and up) or past what your card fits, and Refine Details
  at 4K (these also need the experimental **High-resolution video to video**
  option). 4K IC-LoRA on a card under 32 GB is refused in both modes.
- Retake and Extend at 4K past the 65-frame decode ceiling (~2.5 s at 24 fps)
  or past what your card fits (also needs the experimental
  **High-resolution video to video** option).

## On a Mac (Apple Silicon)

### Mac in 30 seconds

- **Check your macOS version** (Apple menu → About This Mac): local generation
  needs **macOS 26 or later**. The app itself runs on macOS 14 and later. On
  macOS 14 and 15 local generation is unavailable, and the app shows the
  **Coming soon to your device** notice.
- **Find your memory size** (Apple menu → About This Mac): **24 GB runs most
  jobs up to 1080p; 48 GB and up runs it all.** Macs with 16 GB or less, and
  Intel Macs, run local generation via the API instead — the app routes you
  automatically.
- **1440p runs on every Mac, 4K needs 32 GB, and both take hours.** In
  Cautious mode 1440p runs up to 6 seconds at 24 fps on a 24 GB Mac, 10 seconds
  on 32 and 36 GB Macs, and 20 seconds on 48 GB Macs and up. 4K runs up to 2
  seconds on a 32 GB Mac, 3 seconds on a 36 GB Mac and 10 seconds on 48 GB and
  up.
- **Higher frame rates cost more memory.** 25 fps matches 24 fps almost
  everywhere; 48/50 fps render twice the frames, so the longest clips step
  down — 1080p stops at 8 seconds on a 24 GB Mac and 10 seconds on 32–36 GB
  Macs, and 1440p at 3 to 6 seconds below 48 GB.
- **Ultrawide costs more memory.** Every cell offers 21:9 through 9:16; the
  tables show 16:9. Narrower shapes fit wherever 16:9 fits, but 21:9
  ultrawide costs ~30% more and can refuse where 16:9 runs.
- **Ingredients from tall phone photos is the priciest Mac IC-LoRA job.** A
  20-second clip needs a 48 GB Mac (it asks ~29.4 GB against a typical ~25.0 GB
  allowance on 32 GB Macs and ~28.1 GB on 36 GB Macs); wide images run
  everywhere.
- **Closing other apps won't unlock new cells.** The check is against a fixed
  cap for your model, not free memory — though closing apps can still speed
  runs up.
- **Disk space is an install question, not a per-job one.** The model pack
  needs ~72 GB free to download (see the README); generating itself only
  needs room for the output video, so the app never refuses a job over disk.

### How to read the Mac tables

**Columns — your Mac's total memory.** Apple menu → About This Mac. Find
the band closest to your Mac: 24 GB, 32 GB, 36 GB, or 48 GB and up. The tables are
the typical picture — the app checks your exact machine when you press
Generate, and its verdict is final.

**Below 24 GB** (8/16 GB Macs) and **Intel Macs**: local generation is not
offered; the app routes you to the API instead. This is a deliberate floor: we
have not run local generation on Macs under 24 GB, so the app uses the API there.

**Rows — the job you want.** Resolution × length, at 24 fps in 16:9 — the
anchor the generation tables are quoted at. Lengths between two rows fall
between them; the app computes your exact clip when you press Generate. The
frame-rate and aspect-ratio notes under the generation table say how the
other picker values differ.

**Cells — what the app will do:**

| Mark     | Meaning                                                                          |
| -------- | -------------------------------------------------------------------------------- |
| **Runs** | Offered on your Mac in one pass. Always streamed; no Full/Stream split.          |
| **Parts (n)** | Runs in _n_ overlapping time windows. Slower; plain text-to-video can drift a little across parts. |
| **—**    | Not offered. The app tells you why instead of failing halfway.                   |

In the app, a **—** cell shows as a greyed-out option with the reason
attached — the picker tells you before Generate, not after. Frame-rate and
duration pickers are linked: picking 50 fps can grey out long durations that
24 fps offers, each with its reason. Video-editing and IC-LoRA options depend
on your source clip, so the same picker can offer more (or less) once you pick
one. Editing and IC-LoRA follow your source media — they have no frame-rate or
aspect-ratio pickers, so only the generation table carries those notes.

**Closing other apps won't unlock new cells.** Macs share one pool of memory
between everything, so the check is against a fixed cap for your model, not
free memory — though closing apps can still speed runs up. Every job's ask
includes about 3 GB for the buffers the model loader keeps while it streams
the weights.

### Text, image & audio generation (Mac)

Text-to-video, image-to-video, and audio-to-video are all budgeted the same
way on Mac: what counts is **resolution × length**. Editing and IC-LoRA have
their own costs, covered separately below. The table is the 24 fps picture in
16:9 in Cautious mode; the notes below it cover the other frame rates and
shapes.

| Job (resolution × length) | 24 GB Mac | 32 GB Mac | 36 GB Mac | 48 GB Mac and up |
| ------------------------- | --------- | --------- | --------- | ---------------- |
| 540p up to 20 s           | Runs      | Runs      | Runs      | Runs             |
| 720p up to 10 s           | Runs      | Runs      | Runs      | Runs             |
| 720p up to 20 s           | Parts (2) | Runs      | Runs      | Runs             |
| 1080p up to 5 s           | Runs      | Runs      | Runs      | Runs             |
| 1080p up to 10 s          | Parts (3) | Parts (2) | Runs      | Runs             |
| 1080p up to 20 s          | —¹        | Parts (4) | Parts (3) | Parts (3)        |
| 1440p up to 5 s           | Parts (4) | Parts (2) | Runs      | Runs             |
| 1440p up to 10 s          | —²        | Parts (4) | Parts (4) | Parts (2)        |
| 1440p up to 20 s          | —¹        | —¹        | —¹        | Parts (5)³       |
| 4K up to 3 s              | —         | —⁴        | Parts (2)⁴ | Parts (2)       |
| 4K up to 10 s             | —         | —         | —         | Parts (7)⁵       |

¹ More than 5 parts, past the Mac runs: Cautious mode refuses it, and
[Brave mode](#brave-mode) runs it with a warning.
² A 24 GB Mac stops at 6 s (4 parts of 49 frames).
³ We ran 1440p on one M5 Max, up to **5 windows** of 129 frames (20 s at 24 or
25 fps, 10 s at 48 or 50 fps). The app asks ~34.7 GB for it; 24–36 GB Macs and
clips longer than the five-window run are calculated, not run. Like 4K, 20
seconds at 1440p takes hours (about 2 hours on the test Mac).
⁴ A 2-second 4K clip in two short parts asks ~23.0 GB and fits a 32 GB Mac's
~25.0 GB allowance; a 3-second clip asks ~27.5 GB, which needs a 36 GB Mac.
⁵ Runs in overlapping windows of 49 frames, up to **7 windows** — the longest
run we completed (a 10-second clip at 24 fps is exactly 7). It asks ~32.3 GB,
so it needs a 48 GB Mac or larger. 4K takes hours: our test Mac needed close
to three hours for a 10-second clip, and about 25 minutes more for every extra
window. All Mac 4K runs were on that one M5 Max; what smaller Macs need is
calculated from them, not run on those Macs.

Smaller Macs use shorter parts than the 129-frame (1440p) and 49-frame (4K)
windows we ran on the Mac. Those shorter parts are calculated from the Mac runs
and from the Windows runs, which show a part's extra memory scaling with its
size; they have not been run on a Mac.

**Frame rate (24/25/48/50 fps).** 25 fps matches the table except 4K on 48 GB
Macs (8 s instead of 10 s). 48 and 50 fps render twice the frames, so long
clips need more parts, and the 5-part limit sets the top: 540p and 720p run
through 20 s on every Mac; 1080p runs through 8 s on a 24 GB Mac, 10 s on 32
and 36 GB Macs and 20 s on 48 GB and up. At 1440p, 48/50 fps fits 3 s on a 24
GB Mac, 6 s on 32 and 36 GB Macs and 10 s on 48 GB and up. At 4K the 7-window
limit sets the top: a 48 GB Mac or larger gets up to 5 s at 48 fps and 4 s at
50 fps, and smaller Macs no 4K clip at 48/50 fps. The picker lists each frame
rate's durations separately.

**Aspect ratio.** Every cell offers 21:9, 16:9, 3:2, 4:3, 1:1, 4:5, and 9:16.
Narrower and square shapes fit wherever 16:9 fits. 21:9 ultrawide costs ~30%
more and can refuse where 16:9 runs: at 1440p it stops at 2 s on a 24 GB Mac,
8 s on a 32 GB Mac and 10 s on 36 GB and up, and at 1080p at 10 s on a 24 GB
Mac. The app re-checks your exact Mac at Generate and names its numbers. The
model was trained on 16:9 and 9:16; the picker marks the other ratios with a
quality warning.

### IC-LoRA on Mac

Video recipes take your source clip, up to **10 seconds**, as on Windows. The
table shows the longest source clip each Mac runs at 24 fps in Cautious mode.

| Recipe and output         | 24 GB Mac | 32 GB Mac and up |
| ------------------------- | --------- | ---------------- |
| Most recipes, 540p        | 9 s       | 10 s             |
| Most recipes, 720p        | 10 s      | 10 s             |
| Most recipes, 1080p       | 2 s       | 10 s             |
| Layout to Render, 1080p   | 2 s       | 10 s             |
| Restore, 1080p            | 4 s       | 10 s             |
| Refine Details, 1080p     | 3 s       | 10 s             |

Layout to Render runs longer clips in overlapping 97-frame windows, which keeps
its memory flat. Restore runs one 97-frame window (4 s) on a 24 GB Mac; longer
1080p Restore clips run in windows that need a 32 GB Mac.

**Ingredients (from images)** on Mac:

| Job (image shape × length)   | 24 GB Mac | 32 GB Mac | 36 GB Mac | 48 GB Mac and up |
| ---------------------------- | --------- | --------- | --------- | ---------------- |
| Wide image, up to 20 s       | Runs      | Runs      | Runs      | Runs             |
| Tall phone photo, up to 10 s | —¹        | Runs      | Runs      | Runs             |
| Tall phone photo, 20 s       | —         | —²        | —²        | Runs             |

¹ Asks ~19.5 GB against a typical ~18.7 GB allowance on 24 GB Macs, so it is
refused; 8 s fits.
² Asks ~29.4 GB against a typical ~25.0 GB allowance on 32 GB Macs and ~28.1 GB
on 36 GB Macs, so it needs a 48 GB Mac.

**Two-stage** (upscaled output) is offered on Macs at small sizes: 540p and
720p up to about 10 s, 1080p up to about 5 s in a single pass. 5-second clips
at 540p and 720p fit a 24 GB Mac; 720p at 10 s and 1080p at 5 s in one pass
need a 36 GB Mac. Bigger canvases refuse here rather than guessing.

**Above 1080p.** The experimental 1440p and 4K IC-LoRA cells (the
**High-resolution video to video** option in the developer panel) are not
offered in Cautious mode on any Mac: none has been run on one, so every such job
needs [Brave mode](#brave-mode).

One fence applies on every path, any machine: the working grid must fit
inside 1024×576. On the default path that only bites photos taller than
about 1.9:1 — past phone portrait — and because it is the _shape_ that
refuses, downscaling the file won't help; crop to a less extreme shape
instead.

### Editing on Mac

Retake and Extend cost the same on Mac: what counts is **resolution × window
length**. Extend counts your existing footage too — a "10 second" extension on
top of 10 seconds of context is budgeted as ~15 seconds of processing, and
extensions always keep at least ~5 seconds of your clip as context. A clip that
needs more than the table is shortened to the longest window your Mac runs
instead of being refused. Windows are at 24 fps, rounded down to the half second.

| Picture | 24 GB Mac | 32 GB Mac | 36 GB Mac | 48 GB Mac and up |
| ------- | --------- | --------- | --------- | ---------------- |
| 540p    | 20 s      | 20 s      | 20 s      | 20 s             |
| 720p    | 11.5 s    | 20 s      | 20 s      | 20 s             |
| 1080p   | 4.5 s     | 8.5 s     | 10.5 s    | 16 s             |

**Above 1080p.** Retake and Extend cap the picture at 1080p. With the
experimental **High-resolution video to video** option on, larger sources are
kept, but we have not run edits that large on any Mac and edits never run
in windows, so every such job needs [Brave mode](#brave-mode).

### What still needs Brave mode (Mac)

Cautious mode refuses these; [Brave mode](#brave-mode) runs them with a
warning:

- Any text/image/audio clip below 1440p that has to run in more than 5 parts
  (for example 1080p 20 s on a 24 GB Mac).
- 1440p text/image/audio generation past 5 parts: past 6 s on a 24 GB Mac, 20 s
  on 32 and 36 GB Macs, and 20 s at 48/50 fps on 48 GB Macs and up.
- 4K text/image/audio generation past 7 windows, and 4K longer than your Mac
  fits (past 2 s on 32 GB, past 3 s on 36 GB, any 4K on 24 GB). These are over
  the memory allowance, so the warning says the job may fail.
- IC-LoRA at 1440p and 4K (these also need the experimental
  **High-resolution video to video** option).
- Retake and Extend above 1080p.
- Two-stage IC-LoRA beyond the lengths listed in the IC-LoRA section (for
  example 540p past 10 s).

## Worked examples

### On Windows

**"I have an RTX 4070 (12 GB) and 32 GB of RAM. What can I do?"**
Generation: up to 720p at 20 seconds and 1080p at 10 seconds, in parts.
Editing: 540p up to ~16 seconds, 720p up to ~9 seconds, or 1080p up to ~3.5 seconds — on an NVMe
drive these run about 1.5× slower than full speed (normal, not broken).
IC-LoRA needs 48 GB of RAM on the fast-drive path (64 on Standard) — with
that, recipes run 540p clips up to 5 seconds and 720p up to 3 seconds, and
Ingredients runs wide images up to 20 seconds.

**"Can I extend my 1080p clip by 10 seconds on a 4090 (24 GB)?"**
Not in one go — one 1080p extension tops out around 5–6 seconds on 24 GB
(the app names the maximum); split it into two extensions.

**"I want a 20-second video of my character from a phone selfie (tall photo)."**
With Ingredients, on 32 GB: yes. On 24 GB or less: no — it needs ~25.9 GB.
Use a wide photo (20 s runs on 12 GB) or a shorter clip instead (10 s runs on
24 GB, 8 s on 16 GB).

**"Why is local generation missing on my 8 GB card?"**
Because 8 GB cards report under the 9 GB floor, so the app routes you to the
API for everything. On a 10 GB card, text/image generation fits, and IC-LoRA
recipes fit short clips (about 3 seconds at 540p; close other apps first) — but streamed jobs also need
32–64 GB of system RAM depending on the path, which 10 GB-card machines
don't always have. Video editing wants a 12 GB card.

**"I have a 5090 (32 GB) with 96 GB of RAM. Any limits?"**
Almost none up to 1080p. The longest 1080p edits (~15.5 s windows) and every
IC-LoRA job up to 1080p run. At higher resolutions it runs 1440p up to 20 s (including at
48 and 50 fps) and 4K up to 10 s (slowly — about 26 minutes for a 10-second 4K
clip), plus 1440p edits (up to ~8.5 s windows) and short 4K edits (up to ~2.5 s
windows), 1440p IC-LoRA up to about 5
seconds (the whole 10 seconds for Restore and Refine Details), and 4K IC-LoRA
up to about 10 seconds with the experimental option on.
Longer 1440p/4K IC-LoRA clips and longer 4K edits need Brave mode. Big jobs still show
"Stream" — that's the app being careful with your VRAM, not a problem.

**"Can my 4090 (24 GB) make a 10-second 4K video?"**
Yes at 24 or 25 fps: it runs in ten 41-frame windows, so expect a long run, and
the memory need on a 24 GB card is calculated from our 5090 runs, not run on
that card. At 48/50 fps it tops out at 6 seconds. 1440p is available up to 20
seconds at 24/25 fps (and 10 seconds at 48/50 fps); a 20-second 1440p clip at
48/50 fps needs Brave mode.

**"Why is 20 seconds missing at 50 fps?"**
Because 50 fps renders twice the frames of 24 fps, so a clip needs more
memory or more parts. On a 16 GB card 1080p stops at 10 seconds at 50 fps (20
at 24 fps), and on a 10 GB card 540p stops at 4 seconds. Pick 24 fps or a
shorter clip — the picker shows each frame rate's own durations.

### On a Mac

**"I have a MacBook Pro with 48 GB. What can I do?"**
Everything in the Mac tables above, including 1080p at 10 seconds, 1440p up to
20 seconds, and 4K up to 10 seconds (the last two take hours). If your Mac has
24 GB, the only things off the menu in Cautious mode are 1080p past 10
seconds, 1440p past 6 seconds, and 4K — shorter or lower-resolution versions of
the same idea still run, some of them in parts.

**"Can I make a 4K video on my 32 GB Mac?"**
Up to a 2-second clip at 24 or 25 fps (two short parts at ~23.0 GB against a
typical ~25.0 GB allowance). A 3-second clip asks ~27.5 GB, which needs a
36 GB Mac. Anything longer asks more than a 32 GB Mac's memory allowance, so
Cautious mode refuses it. Brave mode will try it anyway, with a warning that it may fail or
stall.

## Why did the app say no?

Every refusal names its reason, with the numbers: what the job needs vs what
the machine has (e.g. "needs 15.8 GB of GPU memory but this card has 12.0 GB";
on a Mac it names your Mac's allowance instead). On Windows, when memory is
merely busy rather than too small, the message says to close other apps and
retry instead — on a Mac, closing apps never unlocks new cells. The common
ones:

| Message means…                       | Try this                                                                                                                                       |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Not enough GPU memory                | Shorter clip, lower resolution or frame rate, or a narrower shape than 21:9. On Windows, closing other GPU apps can also free enough to retry. |
| Not enough system RAM (Windows only) | Close heavy programs; on desktops, more RAM is the real fix.                                                                                   |
| Source image too large (IC-LoRA)     | Downscale the photo — or crop it, if it is extremely tall — and retry.                                                                         |
| Extension too long                   | The message names the maximum — split it into two extensions.                                                                                  |
| Untested on this machine             | Lower the resolution, or turn off Cautious mode ([Brave mode](#brave-mode)) to try it anyway (shows the extrapolated memory estimate).         |

### Brave mode

**Brave mode** is the default. Besides the jobs measured to fit, it lets you
try the rest: jobs that are over your machine's budget, and jobs nobody has
measured on your kind of hardware yet (the unmeasured 1440p and 4K cells
above, and anything past the window counts listed there). The memory meter
shows the risk, and the app asks before running such a job (Settings →
Hardware budget → "Ask before risky generates" turns the question off). The
job may fail, run out of memory, or stall the app. **Cautious mode**
(Settings → Hardware budget) runs only the jobs measured to fit, and refuses
the rest.

Brave mode does not lift hard limits. Jobs the app cannot run at all still
refuse: no local pipeline on the machine, a Mac clip too long for the GPU to
process in one pass, a photo over the spatial limit, or Refine Details on a long
high-resolution clip its decoder cannot fit. For text/image/audio clips that
run in windows, the app still tries every window length first; Brave only
applies when even the smallest window does not fit.

If you believe a refusal is wrong — a job the tables above say should run —
that _is_ a bug we'd like to hear about. Note your machine and memory
(graphics card, RAM, and drive type on Windows; Mac model and memory on a
Mac), and the exact job (resolution, length, frame rate, shape); the last section below says where the numbers come from.

## Where these numbers come from

The app's memory budget comes from running real jobs on a Windows RTX 5090 PC
and an Apple Silicon Mac: finding the exact memory limit of each job (by
deliberately giving it too little), running the same PC with less system RAM to
find the RAM floors, and re-running the longest clips to confirm the top end
really doesn't fit. Between the jobs we ran, the app interpolates, and the
tables on this page are read off that budget.

1440p and 4K are the exception. We ran those on an RTX 5090 (Fast 1440p and 4K,
image and audio to video at 1440p and 4K, Retake and Extend at 1440p and 4K, IC-LoRA at
1440p and 4K) and on one Apple Silicon Mac (Fast 1440p and 4K, with the memory
allowance capped to find each edge). The app offers only up to the number of
windows those runs covered. We did not run other card sizes (16 and 24 GB Windows cards, 24–48 GB Macs); their memory need is calculated from the runs above, which is
why those rows are footnoted as calculated. Anything beyond the runs is a
Brave-mode bet.

For the frame-minded: durations land on a frame grid of 8n + 1 frames, so 5 s at
24 fps is 121 frames — and 50 fps renders about twice the frames of 24 fps,
which is why its cells sit lower.

If what the app does and what these tables say disagree, the tables are out of
date — please report it. The Mac figures on this page include the ~3 GB the
model loader holds while it streams the weights.
