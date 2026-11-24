# Desktop ETA calibration — Apple M5 Max

Measured against `feat/desktop-eta` before `b9ef038`, then used to refit the Darwin coefficients in that commit. 23 local Fast generations, ConvVAE on, cautious budget, LTX 2.5 Fast, streaming from disk. Apple M5 Max, 40-core GPU, 128 GB unified memory. Prompts were sent as already-enhanced so Gemma rewrite time is not in the wall clock.

`spec` is `estimated_seconds` from `GET /api/generate/models-specs` before the job. `est` is `POST /api/generate/estimate`. `start` is the first queue `remainingSeconds`. `wall` is `(finished_at - started_at) / 1000`. Stage splits are from the backend log: load ends at `Running denoising loop`, denoise ends at `Freed resident transformer`, decode is `Building video decoder` to `Video saved to`.

## Runs

| Run | Job | Wall s | Spec s | Est s | Start s | Factor after |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | T2V 270p 3s 16:9 | 33.3 | 58 | 58 | 58 | 0.25 |
| 2 | T2V 270p 3s 16:9 | 33.5 | 37 | 37 | 37 | 0.25 |
| 3 | T2V 360p 4s 16:9 | 60.6 | 45 | 45 | 45 | 0.52 |
| 4 | T2V 540p 5s 16:9 | 164.2 | 112 | 108 | 108 | 0.86 |
| 5 | T2V 270p 2s 16:9 | 32.7 | 47 | 47 | 47 | 0.25 |
| 6 | T2V 270p 5s 16:9 | 65.4 | 42 | 42 | 41 | 0.25 |
| 7 | T2V 360p 3s 16:9 | 59.5 | 41 | 41 | 41 | 0.25 |
| 8 | T2V 360p 6s 16:9 | 115.0 | 52 | 52 | 52 | 0.53 |
| 9 | T2V 540p 3s 16:9 | 114.3 | 82 | 82 | 82 | 0.53 |
| 10 | T2V 540p 8s 16:9 | 332.2 | 166 | 166 | 166 | 1.20 |
| 11 | T2V 540p 10s 16:9 | 407.9 | 418 | 418 | 418 | 1.17 |
| 12 | T2V 720p 5s 16:9 | 297.6 | 317 | 317 | 317 | 1.13 |
| 13 | T2V 720p 10s 16:9 | 655.9 | 640 | 572 | 572 | 1.16 |
| 14 | T2V 1080p 5s 16:9 | 773.0 | 816 | 841 | 841 | 1.09 |
| 15 | T2V 540p 5s 9:16 | 195.8 | 198 | 198 | 198 | 0.25 |
| 16 | T2V 540p 5s 1:1 | 106.6 | 55 | 55 | 54 | 0.25 |
| 17 | T2V 540p 5s 21:9 | 256.1 | 85 | 85 | 85 | 0.33 |
| 18a | T2V 360p 4s 4:3 | 59.4 | 44 | 44 | 44 | 0.25 |
| 18b | T2V 360p 4s 3:2 | 66.7 | 42 | 42 | 42 | 0.25 |
| 19 | T2V 540p 5s 48fps | 400.4 | 114 | 114 | 113 | 0.54 |
| 20 | T2V 1440p 5s 16:9 | 1596.8 | 853 | 853 | 852 | 1.03 |
| 21 | I2V 540p 5s 16:9 | 193.0 | 188 | 181 | 181 | 0.25 |
| 22 | A2V 540p (129 frames) | 207.1 | — | — | 74 | 0.25 |
| 23 | cozy-felt 540p 4s | 174.8 | — | — | 64 | 0.25 |
| 24 | Ingredients | not run | | | | |

Run 24 returned `IC_LORA_HF_AUTH_REQUIRED`. The weights are on disk. Create still requires Hugging Face sign-in.

A2V and cozy-felt have no separate pre-gen ETA in this log. The fast-grid lookup is the wrong surface for those routes. The start countdown above is what the queue actually showed. A2V audio produced 129 frames, not the 121 of a 5-second 24 fps text clip.

1440p / 5s is one window. The first chunk candidate is 129 frames and the clip is 121, so the multi-window check did not apply. No cell was over the cautious budget.

## Calibration did not converge

The cold table, before any run, versus the wall times those cells later measured:

| Cell | Cold spec s | Wall s | Error |
| --- | ---: | ---: | ---: |
| 270p 3s | 58 | 33.3 | +74% |
| 360p 4s | 88 | 60.6 | +45% |
| 540p 5s | 184 | 164.2 | +12% |

Mean absolute error of those three cold specs: **44%**.

After run 1 the factor clamped to the floor, 0.25. Specs became 37 / 45 / 72. Mean absolute error against the same three walls: **31%**. The 270p repeat got closer (37 vs 33.5, +10%). The other two got worse, because a single short clip cannot identify the machine when most of the prediction is a fixed 18 second load.

Run 2 was the same 270p cell and did not move the frontier. The factor stayed 0.25. Run 3 then missed by 26% (45 vs 60.6). Run 4, after the factor had risen to 0.52, missed 540p by 32% (112 vs 164).

Through phase B the live factor swung between the 0.25 floor and about 1.2. Mean absolute error of the pre-gen spec against wall for runs 5–21: **31%**. Long 16:9 cells taken while the factor sat near 1 were close (540p/10s +2%, 720p/5s +7%, 720p/10s −2%, 1080p/5s +6%). The same shapes taken after the factor had fallen back to 0.25 were not (540p/5s at 48 fps −72%, 1440p/5s −47% at factor 0.54).

## Pre-gen spec versus the countdown at start

`spec` and `start` matched within 5 seconds on every 270p and 360p run, and on most 540p runs. They did not on:

| Run | Spec s | Start s | Gap |
| --- | ---: | ---: | ---: |
| 13 720p 10s | 640 | 572 | 68 s |
| 14 1080p 5s | 816 | 841 | 25 s |
| 21 I2V 540p 5s | 188 | 181 | 7 s |

On those three the countdown followed `POST /api/generate/estimate`, not the models-specs cell. The button and the queue were different clocks.

## Stages, old law at factor 1.0

Every run spent 3.5–5.0 seconds from job start to the denoising loop. The old law predicted 18 seconds of load on all of them. That single constant is why short clips looked impossibly fast and the factor pinned at 0.25.

Denoise at factor 1.0 was within about 15% on 540p and above, and high on the shortest 270p and 360p clips (31 s predicted vs 21 s measured for 270p/3s).

Decode was the other systematic miss. One rate for both passes under-predicted 540p–720p decode (540p/10s 137 s predicted vs 194 s measured, 720p/10s 226 vs 288) and, with the old quadratic, over-predicted 1440p (608 vs 567). 1080p decode was closer (281 vs 313).

Log splits match the wall clock within a second on every run except 18a, where the decode marker latched onto the wrong `Video saved to` line. Use the wall time for that row, not the decode split.

## Live countdown

On run 1 the mid-denoise projection was better than the cold spec: `|58 − 33.3| = 25 s` at the start, `|14.9 + 17 − 33.3| = 1.5 s` at step 4/8.

That did not hold once the factor was wrong, and it did not hold on long jobs even when the factor was near 1. The mid projection ran ahead of reality:

| Run | Start error s | Mid projection error s | Decode countdown s | Actual decode s |
| --- | ---: | ---: | ---: | ---: |
| 1 270p 3s | +25 | −1.5 | 9 | 9.3 |
| 11 540p 10s | +10 | −110 | 163 | 195 |
| 12 720p 5s | +19 | −73 | 121 | 135 |
| 14 1080p 5s | +43 | −165 | 333 | 314 |
| 20 1440p 5s | −744 | −862 | 323 | 568 |

1080p was the one long job whose decode countdown matched the remaining decode. Everywhere else the countdown at step N/N was short, because the decode term it was counting down was short. Samples show it decreasing. They do not show it reaching zero and sticking while the decoder was still running. The value was simply too small.

## Aspect, frame rate, other routes

540p / 5s walls: 16:9 164 s, 9:16 196 s, 1:1 107 s, 21:9 256 s. 9:16 has the same latent count as 16:9 and was 19% slower, so an ETA that equates them is wrong on this machine. 1:1 was faster and 21:9 was slower, in the right direction. The live specs for 1:1 (55 s) and 21:9 (85 s) were not, because the factor was on the floor.

48 fps / 5s is 241 frames, the same count as 24 fps / 10s. Walls were 400 s and 408 s. Frame-count scaling holds. The live spec of 114 s did not. It was the floor factor, not the frame model.

I2V 540p/5s took 193 s against the text-to-video 164 s. The spec of 188 s, taken while the factor was near 1, was close. A2V (129 frames) took 207 s and the queue opened at 74 s. Cozy-felt 540p/4s took 175 s and the queue opened at 64 s. Both countdowns were the floor-factor fast estimate, and both were far under the wall.

## Refit

`b9ef038` changes only the Darwin branch of `_stage_time_parts`, evaluated at factor 1.0 on this M5 Max:

| Term | Before | After |
| --- | ---: | ---: |
| Fixed load | 18 s | 4 s |
| Fixed denoise per step | 1.5 s | 0.6 s |
| Denoise per token per step | 0.0016 | 0.0018 |
| Denoise quadratic | 3.6e-8 | 3.6e-8 |
| Decode per token, 1-stage | 0.0065 | 0.0065 |
| Decode per token, 2-stage | 0.0065 | 0.0095 |
| Decode quadratic | 6.5e-8 | 2.5e-8 |

Mean absolute error of the new law against the 23 measured walls, at factor 1.0, is **7%**. A2V at 129 frames is within 1%. I2V is within 1%. Cozy-felt is 10% low (158 s vs 175 s). The cells still outside 15% are the ones the old load constant had distorted the least cleanly: 270p/3s and 360p/4s about +17%, and the first 540p/5s +18%. 720p, 1080p, and 1440p are within 5%.

This is ConvVAE only. DiffVAE was not measured, and both paths share these coefficients.
