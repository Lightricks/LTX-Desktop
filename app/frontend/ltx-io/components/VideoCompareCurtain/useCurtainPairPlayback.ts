import { type RefObject, useCallback, useEffect } from "react";

import { bindCurtainPairPlayback, syncCurtainPairFromAfter } from "./curtainPairPlayback";

export function useCurtainPairPlayback(
  beforeVideoRef: RefObject<HTMLVideoElement | null>,
  afterVideoRef: RefObject<HTMLVideoElement | null>,
  {
    beforeUrl,
    afterUrl,
    enabled = true,
  }: {
    beforeUrl: string | undefined;
    afterUrl: string | undefined;
    enabled?: boolean;
  },
) {
  useEffect(() => {
    if (!enabled || !beforeUrl || !afterUrl) return;
    const beforeVideo = beforeVideoRef.current;
    const afterVideo = afterVideoRef.current;
    if (!beforeVideo || !afterVideo) return;
    return bindCurtainPairPlayback(beforeVideo, afterVideo);
  }, [afterUrl, afterVideoRef, beforeUrl, beforeVideoRef, enabled]);

  return useCallback(() => {
    syncCurtainPairFromAfter(beforeVideoRef.current, afterVideoRef.current);
  }, [afterVideoRef, beforeVideoRef]);
}
