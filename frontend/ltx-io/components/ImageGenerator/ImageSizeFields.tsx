import { Text } from "@ds/Text/Text";

import { IMAGE_RESOLUTIONS, type ImageResolution } from "@/lib/genspace-image-resolution";

import { FormOptionControl } from "../../screens/Feature/fields/FormOptionControl";
import { getVideoOptionIcon } from "../../screens/Feature/fields/videoOptionIcons";
import type { FieldOption } from "../../screens/Feature/types";
import styles from "./ImageGenerator.module.scss";
import {
  IMAGE_ASPECTS,
  IMAGE_MAX_SIDE,
  type ImageAspect,
  imageResolutionFits,
} from "./imageGeneratorRequest";

/** The shape and resolution pickers. They are the same pickers as in the feature forms. */
export function ImageSizeFields({
  aspect,
  resolution,
  onAspectChange,
  onResolutionChange,
}: {
  aspect: ImageAspect;
  resolution: ImageResolution;
  onAspectChange: (aspect: ImageAspect) => void;
  onResolutionChange: (resolution: ImageResolution) => void;
}) {
  const aspectOptions: FieldOption<ImageAspect>[] = IMAGE_ASPECTS.map((value) => ({
    value,
    label: value,
  }));
  const resolutionOptions: FieldOption<ImageResolution>[] = IMAGE_RESOLUTIONS.map((value) => ({
    value,
    label: value,
    disabledReason: imageResolutionFits(aspect, value)
      ? undefined
      : `Over ${IMAGE_MAX_SIDE} px on the long side at this shape`,
  }));

  return (
    <>
      <div className={styles.sizeField}>
        <Text as="span" variant="body" size="md" className={styles.label}>
          Aspect ratio
        </Text>
        <FormOptionControl
          options={aspectOptions}
          value={aspect}
          onChange={onAspectChange}
          ariaLabel="Aspect ratio"
          forcePicker
          showListIcons
          getOptionIcon={(option) =>
            getVideoOptionIcon({ dataKey: "aspectRatio" }, option)
          }
        />
      </div>
      <div className={styles.sizeField}>
        <Text as="span" variant="body" size="md" className={styles.label}>
          Resolution
        </Text>
        <FormOptionControl
          options={resolutionOptions}
          value={resolution}
          onChange={onResolutionChange}
          ariaLabel="Resolution"
          forcePicker
          getOptionIcon={(option) =>
            getVideoOptionIcon({ dataKey: "resolution" }, option)
          }
        />
      </div>
    </>
  );
}
