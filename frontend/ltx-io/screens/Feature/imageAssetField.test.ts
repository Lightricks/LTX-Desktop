import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { ApiResultError } from "../../lib/unwrapApiResult.ts";
import {
  COULD_NOT_ADD_IMAGE_MESSAGE,
  DROPPED_FILE_PATH_UNREADABLE_MESSAGE,
  IMAGE_ACCEPT,
  IMAGE_EXTENSIONS,
  UNAVAILABLE_IMAGE_MESSAGE,
  UNSUPPORTED_IMAGE_MESSAGE,
  imageIngestErrorMessage,
  imageLookupErrorMessage,
  isSupportedDroppedImage,
  prepareDroppedImageIngest,
} from "./fields/imageAssetInput.ts";
import {
  applyFeatureFieldChange,
  resolveFeatureSeedValues,
  toDurableAssetRef,
  type AssetRef,
  type FeatureDefinition,
  type FeatureFormModel,
  type FeatureValues,
} from "./types.ts";

type ImageAssetSchema = {
  caption: { kind: "textarea" };
  image: { kind: "image-asset"; value: AssetRef | null };
};

type ImageAssetValues = FeatureValues<ImageAssetSchema>;

type ImageAssetBody = {
  caption: string;
  startFrame: AssetRef | null;
};

const defaults: ImageAssetValues = {
  caption: "",
  image: null,
};

const seedAsset: AssetRef = { assetId: "seed-asset" };

const imageAssetDefinition: FeatureDefinition<
  ImageAssetSchema,
  ImageAssetBody,
  { seed: AssetRef | null }
> = {
  id: "text-to-video",
  title: "Synthetic image asset",
  defaults,
  initialValues: (context) => ({
    caption: "",
    image: context.seed,
  }),
  form: (): FeatureFormModel<ImageAssetSchema> => ({
    fields: [
      {
        kind: "image-asset",
        id: "image",
        label: "Image",
        dataKey: "image",
      },
      {
        kind: "textarea",
        id: "caption",
        label: "Caption",
        dataKey: "caption",
      },
    ],
  }),
  applyChange: (values, change) => applyFeatureFieldChange(values, change),
  normalize: (values) => values,
  validate: (values) =>
    values.image
      ? []
      : [
          {
            id: "image-required",
            fieldId: "image",
            message: "Image is required.",
          },
        ],
  fromGeneration: (spec) => {
    if (
      typeof spec !== "object" ||
      spec === null ||
      !("image" in spec) ||
      typeof spec.image !== "object" ||
      spec.image === null ||
      !("assetId" in spec.image) ||
      typeof spec.image.assetId !== "string"
    ) {
      return null;
    }
    return { caption: "", image: { assetId: spec.image.assetId } };
  },
  toCreateBody: (values) => ({
    caption: values.caption,
    startFrame: values.image,
  }),
};

describe("image-asset Feature field", () => {
  it("builds a generic image-asset field whose durable value is AssetRef or null", () => {
    const form = imageAssetDefinition.form(defaults, { seed: seedAsset });

    assert.deepEqual(
      form.fields.map(({ kind, id, dataKey }) => ({ kind, id, dataKey })),
      [
        { kind: "image-asset", id: "image", dataKey: "image" },
        { kind: "textarea", id: "caption", dataKey: "caption" },
      ],
    );
  });

  it("stores only assetId when applying an image-asset change", () => {
    const next = imageAssetDefinition.applyChange(
      defaults,
      { kind: "image-asset", fieldId: "image", dataKey: "image", value: { assetId: "uploaded-1" } },
      { seed: seedAsset },
    );

    assert.deepEqual(next.image, { assetId: "uploaded-1" });
    assert.equal(next.image && "path" in next.image, false);
  });

  it("accepts a cleared null image without restoring a file path", () => {
    const filled: ImageAssetValues = {
      caption: "still",
      image: { assetId: "uploaded-1" },
    };
    const next = imageAssetDefinition.applyChange(
      filled,
      { kind: "image-asset", fieldId: "image", dataKey: "image", value: null },
      { seed: seedAsset },
    );

    assert.equal(next.image, null);
  });

  it("keeps only assetId when converting an ingested asset to a feature value", () => {
    const ingested = {
      id: "asset-9",
      path: "/Users/me/Movies/input.jpg",
      name: "input.jpg",
    };
    const ref = toDurableAssetRef(ingested);

    assert.deepEqual(ref, { assetId: "asset-9" });
    assert.deepEqual(Object.keys(ref), ["assetId"]);
  });
});

describe("FeatureDefinition initialValues seed seam", () => {
  it("seeds from initialValues when the form has never been persisted", () => {
    const seeded = resolveFeatureSeedValues(imageAssetDefinition, {
      context: { seed: seedAsset },
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: undefined,
    });

    assert.deepEqual(seeded, { caption: "", image: seedAsset });
  });

  it("does not reinsert a seed after the user persists a cleared null", () => {
    const cleared: ImageAssetValues = { caption: "kept", image: null };
    const next = resolveFeatureSeedValues(imageAssetDefinition, {
      context: { seed: seedAsset },
      hasStoredValues: true,
      storedValues: cleared,
      lastGenerationSpec: undefined,
    });

    assert.deepEqual(next, cleared);
    assert.equal(next.image, null);
  });

  it("prefers a persisted asset over the seed", () => {
    const stored: ImageAssetValues = {
      caption: "user",
      image: { assetId: "user-asset" },
    };
    const next = resolveFeatureSeedValues(imageAssetDefinition, {
      context: { seed: seedAsset },
      hasStoredValues: true,
      storedValues: stored,
      lastGenerationSpec: undefined,
    });

    assert.deepEqual(next, stored);
  });

  it("hydrates from a generation before applying the seed", () => {
    const next = resolveFeatureSeedValues(imageAssetDefinition, {
      context: { seed: seedAsset },
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: { image: { assetId: "from-history" } },
    });

    assert.deepEqual(next, { caption: "", image: { assetId: "from-history" } });
  });
});

describe("dropped image prefiltering", () => {
  it("does not invoke the ingest path for a dropped video", () => {
    let pathLookups = 0;
    const result = prepareDroppedImageIngest(
      { name: "clip.mp4", type: "video/mp4" },
      () => {
        pathLookups += 1;
        return "/tmp/clip.mp4";
      },
    );

    assert.deepEqual(result, {
      error: "This field accepts images only.",
    });
    assert.equal(pathLookups, 0);
  });

  it("does not invoke the ingest path for a dropped pdf", () => {
    let pathLookups = 0;
    const result = prepareDroppedImageIngest(
      { name: "notes.pdf", type: "application/pdf" },
      () => {
        pathLookups += 1;
        return "/tmp/notes.pdf";
      },
    );

    assert.equal("ingestPath" in result, false);
    assert.equal(pathLookups, 0);
  });

  it("does not invoke the ingest path when a known non-image MIME has an image extension", () => {
    let pathLookups = 0;
    const result = prepareDroppedImageIngest(
      { name: "clip.jpg", type: "video/mp4" },
      () => {
        pathLookups += 1;
        return "/tmp/clip.jpg";
      },
    );

    assert.equal("ingestPath" in result, false);
    assert.equal(pathLookups, 0);
  });

  it("does not invoke the ingest path for a dropped gif", () => {
    let pathLookups = 0;
    const result = prepareDroppedImageIngest(
      { name: "loop.gif", type: "image/gif" },
      () => {
        pathLookups += 1;
        return "/tmp/loop.gif";
      },
    );

    assert.deepEqual(result, {
      error: "This field accepts images only.",
    });
    assert.equal(pathLookups, 0);
  });

  it("does not invoke the ingest path for a gif whose MIME is missing", () => {
    let pathLookups = 0;
    const result = prepareDroppedImageIngest(
      { name: "loop.GIF", type: "" },
      () => {
        pathLookups += 1;
        return "/tmp/loop.GIF";
      },
    );

    assert.equal("ingestPath" in result, false);
    assert.equal(pathLookups, 0);
  });

  it("advertises only backend-supported image types", () => {
    assert.deepEqual([...IMAGE_EXTENSIONS], ["png", "jpg", "jpeg", "webp"]);
    assert.equal(IMAGE_ACCEPT.includes("gif"), false);
    assert.equal(IMAGE_ACCEPT.includes("image/jpeg"), true);
    assert.equal(IMAGE_ACCEPT.includes("image/png"), true);
    assert.equal(IMAGE_ACCEPT.includes("image/webp"), true);
  });

  it("resolves a dropped jpeg for ingest", () => {
    const result = prepareDroppedImageIngest(
      { name: "still.jpg", type: "image/jpeg" },
      () => "/Users/me/still.jpg",
    );

    assert.deepEqual(result, { ingestPath: "/Users/me/still.jpg" });
  });

  it("resolves dropped png and webp for ingest", () => {
    assert.deepEqual(
      prepareDroppedImageIngest(
        { name: "still.png", type: "image/png" },
        () => "/tmp/still.png",
      ),
      { ingestPath: "/tmp/still.png" },
    );
    assert.deepEqual(
      prepareDroppedImageIngest(
        { name: "still.webp", type: "image/webp" },
        () => "/tmp/still.webp",
      ),
      { ingestPath: "/tmp/still.webp" },
    );
    assert.deepEqual(
      prepareDroppedImageIngest(
        { name: "still.jpeg", type: "" },
        () => "/tmp/still.jpeg",
      ),
      { ingestPath: "/tmp/still.jpeg" },
    );
  });

  it("allows a legitimate image whose browser MIME is missing when the extension is allowlisted", () => {
    const result = prepareDroppedImageIngest(
      { name: "photo.PNG", type: "" },
      () => "/tmp/photo.PNG",
    );

    assert.deepEqual(result, { ingestPath: "/tmp/photo.PNG" });
  });

  it("rejects a missing MIME when the extension is not an allowlisted image", () => {
    let pathLookups = 0;
    const result = prepareDroppedImageIngest(
      { name: "notes.txt", type: "" },
      () => {
        pathLookups += 1;
        return "/tmp/notes.txt";
      },
    );

    assert.equal("ingestPath" in result, false);
    assert.equal(pathLookups, 0);
  });
});

describe("image lookup error copy", () => {
  it("maps ASSET_NOT_FOUND to a generic unavailable-image message", () => {
    const message = imageLookupErrorMessage(
      new ApiResultError("ASSET_NOT_FOUND", {
        code: "ASSET_NOT_FOUND",
        status: 404,
      }),
    );

    assert.equal(message, "This image is no longer available.");
    assert.equal(message.includes("ASSET_NOT_FOUND"), false);
  });

  it("preserves meaningful errors for other lookup failures", () => {
    assert.equal(
      imageLookupErrorMessage(
        new ApiResultError("STORE_UNAVAILABLE", {
          code: "STORE_UNAVAILABLE",
          status: 503,
        }),
      ),
      "STORE_UNAVAILABLE",
    );
    assert.equal(
      imageLookupErrorMessage(new Error("Could not load this image.")),
      "Could not load this image.",
    );
  });
});

describe("image ingest error copy", () => {
  it("maps FILE_NOT_FOUND without leaking the filesystem path", () => {
    const seedPath =
      "/Applications/LTX.app/Contents/Resources/seeds/start-frame.png";
    const message = imageIngestErrorMessage(
      new ApiResultError(`file not found: ${seedPath}`, {
        code: "FILE_NOT_FOUND",
        status: 400,
      }),
    );

    assert.equal(message, UNAVAILABLE_IMAGE_MESSAGE);
    assert.equal(message.includes(seedPath), false);
    assert.equal(message.includes("file not found"), false);
  });

  it("maps UNSUPPORTED_MEDIA without leaking the source path", () => {
    const source = "/Users/me/Movies/clip.gif";
    const message = imageIngestErrorMessage(
      new ApiResultError(`unsupported media type for ${source}`, {
        code: "UNSUPPORTED_MEDIA",
        status: 400,
      }),
    );

    assert.equal(message, UNSUPPORTED_IMAGE_MESSAGE);
    assert.equal(message.includes(source), false);
    assert.equal(message.includes("unsupported media type"), false);
  });

  it("uses generic copy for other ingest failures, including path-bearing messages", () => {
    const oversized = "/tmp/huge.png";
    const oversizedMessage = imageIngestErrorMessage(
      new ApiResultError(`file too large: ${oversized}`, {
        code: "FILE_TOO_LARGE",
        status: 400,
      }),
    );
    const uncodedMessage = imageIngestErrorMessage(
      new Error(`file not found: ${oversized}`),
    );

    assert.equal(oversizedMessage, COULD_NOT_ADD_IMAGE_MESSAGE);
    assert.equal(uncodedMessage, COULD_NOT_ADD_IMAGE_MESSAGE);
    assert.equal(imageIngestErrorMessage({}), COULD_NOT_ADD_IMAGE_MESSAGE);
    assert.equal(
      imageIngestErrorMessage(new Error(DROPPED_FILE_PATH_UNREADABLE_MESSAGE)),
      DROPPED_FILE_PATH_UNREADABLE_MESSAGE,
    );
    assert.equal(oversizedMessage.includes(oversized), false);
    assert.equal(uncodedMessage.includes(oversized), false);
  });
});

describe("browse path allowlist", () => {
  it("rejects a gif selected via OS All files before ingest", () => {
    assert.equal(
      isSupportedDroppedImage({
        name: "/Users/me/Pictures/loop.gif",
        type: "",
      }),
      false,
    );
    assert.equal(
      isSupportedDroppedImage({
        name: "/Applications/LTX.app/Contents/Resources/seeds/loop.GIF",
        type: "",
      }),
      false,
    );
  });

  it("rejects an unknown suffix selected via OS All files before ingest", () => {
    assert.equal(
      isSupportedDroppedImage({ name: "/tmp/notes.txt", type: "" }),
      false,
    );
    assert.equal(
      isSupportedDroppedImage({ name: "/tmp/still.xyz", type: "" }),
      false,
    );
  });

  it("accepts allowlisted image suffixes when MIME is empty", () => {
    assert.equal(
      isSupportedDroppedImage({ name: "/Users/me/still.jpg", type: "" }),
      true,
    );
    assert.equal(
      isSupportedDroppedImage({ name: "/tmp/still.WEBP", type: "" }),
      true,
    );
  });
});
