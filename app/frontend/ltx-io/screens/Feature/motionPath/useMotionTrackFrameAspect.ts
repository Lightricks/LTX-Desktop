import { useEffect, useState } from "react";

import { motionTrackFrameAspect } from "./motionPathPreviewFrameStyle";

/** The width / height of the frame the pipeline renders for the image, or null until it loads. */
export function useMotionTrackFrameAspect(imageUrl: string | undefined): number | null {
  const [aspect, setAspect] = useState<number | null>(null);

  useEffect(() => {
    setAspect(null);
    if (!imageUrl) return;
    let isCurrent = true;
    const image = new Image();
    image.onload = () => {
      if (isCurrent && image.naturalWidth > 0 && image.naturalHeight > 0) {
        setAspect(motionTrackFrameAspect(image.naturalWidth, image.naturalHeight));
      }
    };
    image.src = imageUrl;
    return () => {
      isCurrent = false;
      image.onload = null;
    };
  }, [imageUrl]);

  return aspect;
}
