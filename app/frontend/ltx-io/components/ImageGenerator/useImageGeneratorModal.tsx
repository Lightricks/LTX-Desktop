import { useCallback } from "react";

import type { ExploreAsset } from "@/lib/explore-contract";
import { useCardModal } from "@/ltx-io/components/shared/Modal/useCardModal";

import { ImageGenerator } from "./ImageGenerator";
import type { ImageAspect } from "./imageGeneratorRequest";

export const IMAGE_GENERATOR_MODAL_NAME = "image-generator";

/**
 * Opens the image generator in a modal. `onUseImage` receives the finished image. The modal
 * closes after it returns.
 */
export function useImageGeneratorModal(): (input: {
  initialPrompt?: string;
  initialAspect?: ImageAspect;
  onUseImage?: (asset: ExploreAsset) => void;
  useImageLabel?: string;
}) => void {
  const showCardModal = useCardModal(IMAGE_GENERATOR_MODAL_NAME, { closeButton: true });

  return useCallback(
    ({ initialPrompt, initialAspect, onUseImage, useImageLabel }) => {
      showCardModal(({ apply, cancel, setShouldPreventDismiss }) => (
        <ImageGenerator
          initialPrompt={initialPrompt}
          initialAspect={initialAspect}
          useImageLabel={useImageLabel}
          onCloseModal={cancel}
          setShouldPreventDismiss={setShouldPreventDismiss}
          onUseImage={
            onUseImage
              ? (asset) => {
                  apply();
                  onUseImage(asset);
                }
              : undefined
          }
        />
      ));
    },
    [showCardModal],
  );
}
