import assert from "node:assert/strict";
import { test } from "node:test";
import { specFixture, specFixtureBothOfferings } from "./specFixture.ts";
import { textToVideoDefinition as t2v } from "./textToVideo.ts";
import { imageToVideoDefinition as i2v } from "./imageToVideo.ts";
import { hasManagedVideoModels, type AmdPickerContext } from "../../../lib/amdVideoPicker.ts";

function context() {
  const specs = specFixture();
  const row = specs.downloaded_local_models![0]!;
  specs.comfy_local_model = row;
  specs.kandinsky_local_model = { ...row, spec: {
    display_name: "Kandinsky 6 Lite Distill",
    supported_resolutions_durations: { "540p": { fps_to_durations: { "24": [5] }, aspect_ratios: ["16:9", "9:16"] } },
  }};
  const amd: AmdPickerContext = { preferred: "comfy", status: "ready", message: "", kandinskyReady: true, kandinskyMessage: "", desktop: true };
  return { specs, amd, seed: { assetId: "start" } };
}

test("Comfy-only installs can queue INT8 without a native LTX checkpoint", () => {
  const ctx = context();
  ctx.specs.downloaded_local_models = [];
  const values = t2v.initialValues!(ctx);
  assert.equal(values.model, "comfy");
  assert.equal(t2v.isReady!(values, ctx), true);
  const field = t2v.form(values, ctx).fields.find(f => f.id === "negativePrompt");
  assert.ok(field?.kind === "textarea" && field.disabledReason?.includes("positive prompt"));
  const body = t2v.toCreateBody(values, ctx);
  assert.equal(body.params.amdFastEngine, "comfy");
  assert.equal(body.params.negativePrompt, "");
  assert.equal(t2v.fromGeneration(body, ctx)?.model, "comfy");
  assert.equal(t2v.isReady!(values, {...ctx, amd: {...ctx.amd, status:"missing"}}), false);
  const model = t2v.form(values,ctx).fields.find(f => f.id==="model");
  assert.ok(model?.kind === "options");
  assert.equal(model.options.filter(o => o.disabledReason).length, 2);
});

test("choosing native explicitly overrides the AMD default and retains the typed negative prompt", () => {
  const ctx = context();
  const values = t2v.applyChange!(t2v.initialValues!(ctx),
    { kind:"options", fieldId:"model", dataKey:"model", value:"ltx-2.5-fast" },ctx);
  const body = t2v.toCreateBody(values,ctx);
  assert.equal(body.params.amdFastEngine, "native");
  assert.equal(body.params.negativePrompt, values.negativePrompt);
  assert.equal(t2v.fromGeneration(body,ctx)?.model,"ltx-2.5-fast");
});

test("Kandinsky clamps the shape and duration and restores the renderer on a rerun", () => {
  const ctx = context();
  const previous = { ...i2v.initialValues!(ctx), duration: 20 as const, fps:48 as const, resolution:"1080p" as const };
  const values = i2v.applyChange!(previous, {kind:"options", fieldId:"model", dataKey:"model", value:"kandinsky"},ctx);
  assert.equal(values.model,"kandinsky");
  assert.equal(values.duration,5);
  assert.equal(values.fps,24);
  assert.equal(i2v.isReady!(values,ctx),true);
  assert.equal(i2v.isReady!(values,{...ctx,amd:{...ctx.amd,kandinskyReady:false}}),false);
  const body = i2v.toCreateBody(values,ctx);
  assert.equal(body.params.amdFastEngine,"kandinsky");
  assert.equal(body.params.aspectRatio,"auto");
  assert.equal(i2v.fromGeneration(body,ctx)?.model,"kandinsky");
  const res = i2v.form(values,ctx).fields.find(f=>f.id==="resolution");
  assert.ok(res?.kind==="options");
  assert.deepEqual(res.options.map(o=>o.label), ["480p SD"]);
  assert.ok(i2v.validate({...values,endFrame:{assetId:"end"}},ctx).some(v=>v.fieldId==="endFrame"));
});

test("a checkpoint in the catalog still offers native setup until all required files are installed", () => {
  const ctx = context();
  ctx.amd.native = { "ltx-2.3-fast": "missing", "ltx-2.5-fast": "missing" };
  const values = { ...t2v.initialValues!(ctx), model: "ltx-2.5-fast" as const };
  const model = t2v.form(values, ctx).fields.find(field => field.id === "model");
  assert.ok(model?.kind === "options");
  for (const family of ["ltx-2.3", "ltx-2.5"] as const) {
    const option = model.options.find(row => row.value === `${family}-fast`);
    assert.ok(option);
    assert.deepEqual(option.setup, { kind: family, label: "Download & use" });
    assert.match(option.disabledReason!, /files are missing/);
  }
  assert.equal(t2v.isReady!(values, ctx), false);
  ctx.amd.native["ltx-2.5-fast"] = "ready";
  assert.equal(t2v.isReady!(values, ctx), true);
  ctx.amd.native["ltx-2.3-fast"] = "ready";
  const refreshed = t2v.form(values, ctx).fields.find(field => field.id === "model");
  assert.ok(refreshed?.kind === "options");
  const incompatible = refreshed.options.find(row => row.value === "ltx-2.3-fast");
  assert.equal(incompatible?.setup, undefined);
  assert.match(incompatible?.disabledReason ?? "", /compatible settings/);
});

test("LTX and Kandinsky offer setup and download progress independently in both generation forms", () => {
  const ctx = context();
  ctx.amd.kandinskyReady = false;
  ctx.amd.kandinskyDownloading = true;
  const t = t2v.form(t2v.initialValues!(ctx), ctx).fields.find(field => field.id === "model");
  const i = i2v.form(i2v.initialValues!(ctx), ctx).fields.find(field => field.id === "model");
  for (const field of [t, i]) {
    assert.ok(field?.kind === "options");
    assert.deepEqual(field.options.find(row => row.value === "kandinsky")?.setup,
      { kind: "kandinsky", label: "View download" });
    assert.equal(field.options.find(row => row.value === "comfy")?.setup, undefined);
  }
  ctx.amd.status = "offline";
  const offline = i2v.form(i2v.initialValues!(ctx), ctx).fields.find(field => field.id === "model");
  assert.ok(offline?.kind === "options");
  assert.deepEqual(offline.options.find(row => row.value === "comfy")?.setup, { kind: "comfy-int8", label: "Set up" });
});

test("a downloaded inactive native family offers activation without a download", () => {
  const ctx = context();
  Object.assign(ctx.specs, specFixtureBothOfferings());
  ctx.amd.native = { "ltx-2.3-fast": "ready", "ltx-2.5-fast": "ready" };
  ctx.amd.activeNative = "ltx-2.5-fast";
  const values = t2v.initialValues!(ctx);
  const model = t2v.form(values, ctx).fields.find(field => field.id === "model");
  assert.ok(model?.kind === "options");
  const inactive = model.options.find(row => row.value === "ltx-2.3-fast");
  assert.deepEqual(inactive?.setup, { kind: "ltx-2.3", label: "Activate" });
  assert.equal(inactive?.disabledReason, undefined);
  assert.equal(model.options.find(row => row.value === "ltx-2.5-fast")?.setup, undefined);
  ctx.amd.desktop = false;
  ctx.amd.status = "missing";
  const paired = t2v.form(values, ctx).fields.find(field => field.id === "model");
  assert.ok(paired?.kind === "options");
  assert.equal(paired.options.find(row => row.value === "ltx-2.3-fast")?.setup, undefined);
  assert.deepEqual(paired.options.find(row => row.value === "comfy")?.setup,
    { kind: "comfy-int8", label: "Set up on PC" });
});

function h3Context() {
  const ctx = context();
  const res = { fps_to_durations: { "24": [5, 10, 15] }, aspect_ratios: ["16:9", "9:16", "1:1"] } as const;
  const row = { ...ctx.specs.comfy_local_model!, spec: { display_name: "MiniMax H3", supported_resolutions_durations: {
    "540p": { fps_to_durations: { "24": [...res.fps_to_durations["24"]] }, aspect_ratios: [...res.aspect_ratios] },
    "720p": { fps_to_durations: { "24": [...res.fps_to_durations["24"]] }, aspect_ratios: [...res.aspect_ratios] },
  } } };
  ctx.specs.h3_fl2va_local_model = ctx.specs.h3_fl2va_distill_local_model = ctx.specs.h3_ref2va_local_model = row;
  ctx.amd.h3 = {
    "h3-fl2va": { status: "ready", message: "" }, "h3-fl2va-distill": { status: "downloading", message: "Downloading turbo adapter" },
    "h3-ref2va": { status: "ready", message: "" },
  };
  return ctx;
}

test("H3 base remains available while its turbo adapter downloads and Ref2VA requires the image workflow", () => {
  const ctx = h3Context();
  const values = t2v.initialValues!(ctx);
  const text = t2v.form(values, ctx).fields.find(field => field.id === "model");
  const image = i2v.form(i2v.initialValues!(ctx), ctx).fields.find(field => field.id === "model");
  assert.ok(text?.kind === "options" && image?.kind === "options");
  assert.equal(text.options.find(row => row.value === "h3-fl2va")?.disabledReason, undefined);
  assert.deepEqual(text.options.find(row => row.value === "h3-fl2va-distill")?.setup, { kind: "h3-fl2va-distill", label: "View download" });
  assert.equal(text.options.some(row => row.value === "h3-ref2va"), false);
  assert.ok(image.options.some(row => row.value === "h3-ref2va"));
  const selected = t2v.applyChange!(values, { kind: "options", fieldId: "model", dataKey: "model", value: "h3-fl2va" }, ctx);
  const sizes = t2v.form(selected, ctx).fields.find(field => field.id === "resolution");
  assert.ok(sizes?.kind === "options");
  assert.deepEqual(sizes.options.map(row => row.label), ["480p", "768p"]);
  assert.equal(t2v.toCreateBody(selected, ctx).params.amdFastEngine, "h3-fl2va");
});

test("H3 retains an end-only FL keyframe and restores a 15-second H3 draft", () => {
  const ctx = h3Context();
  const values = { ...i2v.initialValues!(ctx), model: "h3-fl2va" as const, duration: 15 as const,
    startFrame: null, endFrame: { assetId: "end" } };
  assert.deepEqual(i2v.validate(values, ctx), []);
  const body = i2v.toCreateBody(values, ctx);
  assert.equal(body.inputs.startFrame, null);
  assert.deepEqual(body.inputs.endFrame, { assetId: "end" });
  const restored = i2v.fromGeneration(body, ctx);
  assert.equal(restored?.model, "h3-fl2va");
  assert.equal(restored?.duration, 15);
  assert.equal(restored?.startFrame, null);
  assert.deepEqual(restored?.endFrame, { assetId: "end" });
});

test("Ref2VA sends the reference images in order and omits the draft's hidden end keyframe", () => {
  const ctx = h3Context();
  const refs = [{ assetId: "two" }, { assetId: "three" }];
  const values = { ...i2v.initialValues!(ctx), model: "h3-ref2va" as const, endFrame: { assetId: "unused" }, referenceImages: refs };
  assert.deepEqual(i2v.validate(values, ctx), []);
  const fields = i2v.form(values, ctx).fields;
  assert.equal(fields.find(row => row.id === "startFrame")?.label, "Reference Image 1");
  assert.equal(fields.some(row => row.id === "endFrame"), false);
  const body = i2v.toCreateBody(values, ctx);
  assert.deepEqual(body.inputs, { startFrame: { assetId: "start" }, endFrame: null, referenceImages: refs });
  assert.deepEqual(i2v.fromGeneration(body, ctx)?.referenceImages, refs);
});

test("a fresh NVIDIA native default stays selected while ComfyUI, Kandinsky and H3 offer setup", () => {
  const ctx = h3Context();
  ctx.amd.preferred = "native";
  ctx.specs.downloaded_local_models = [];
  ctx.amd.native = { "ltx-2.5-fast": "missing", "ltx-2.3-fast": "missing" };
  ctx.amd.status = "missing";
  ctx.amd.kandinskyReady = false;
  ctx.amd.h3 = {};
  const values = i2v.initialValues!(ctx);
  assert.equal(values.model, "ltx-2.5-fast");
  assert.equal(i2v.isReady!(values, ctx), false);
  const field = i2v.form(values, ctx).fields.find(f => f.id === "model");
  assert.ok(field?.kind === "options");
  assert.deepEqual(field.options.map(option => option.setup?.kind),
    ["comfy-int8", "ltx-2.5", "ltx-2.3", "kandinsky", "h3-fl2va", "h3-fl2va-distill", "h3-ref2va"]);
  assert.ok(field.options.every(option => option.disabledReason));
  ctx.amd.status = "ready";
  const selected = i2v.applyChange!(values, { kind: "options", fieldId: "model", dataKey: "model", value: "comfy" }, ctx);
  assert.equal(i2v.isReady!(selected, ctx), true);
  const saved = i2v.toCreateBody(selected, ctx);
  assert.equal(saved.params.amdFastEngine, "comfy");
  assert.equal(i2v.fromGeneration(saved, ctx)?.model, "comfy");
});

test("a host advertising only H3 Turbo exposes that download without inventing other managed models", () => {
  const ctx = h3Context();
  const specs = specFixture();
  assert.equal(hasManagedVideoModels(specs), false);
  specs.h3_fl2va_distill_local_model = ctx.specs.h3_fl2va_distill_local_model;
  assert.equal(hasManagedVideoModels(specs), true);
  ctx.specs = specs;
  const values = t2v.initialValues!(ctx);
  assert.equal(values.model, "ltx-2.5-fast");
  const field = t2v.form(values, ctx).fields.find(f => f.id === "model");
  assert.ok(field?.kind === "options");
  assert.deepEqual(field.options.map(option => option.value), ["ltx-2.5-fast", "ltx-2.3-fast", "h3-fl2va-distill"]);
});

test("reopening a managed job on a native-only host keeps its prompt and offers installed native models", () => {
  const managed = context();
  const saved = t2v.toCreateBody(t2v.initialValues!(managed), managed);
  const nativeOnly = { specs: specFixture() };
  const restored = t2v.fromGeneration(saved, nativeOnly);
  assert.ok(restored);
  assert.equal(restored.prompt, saved.params.prompt);
  assert.equal(restored.model, "comfy");
  assert.equal(t2v.isReady!(restored, nativeOnly), false);
  const field = t2v.form(restored, nativeOnly).fields.find(field => field.id === "model");
  assert.ok(field?.kind === "options");
  assert.ok(field.options.some(option => option.value === "ltx-2.5-fast" && !option.disabledReason));
  assert.ok(!field.options.some(option => option.value === "comfy"));
  const selected = t2v.applyChange!(restored,
    { kind: "options", fieldId: "model", dataKey: "model", value: "ltx-2.5-fast" }, nativeOnly);
  assert.equal(t2v.isReady!(selected, nativeOnly), true);
  assert.equal(t2v.toCreateBody(selected, nativeOnly).params.amdFastEngine, undefined);
});
