import { useEffect, useState } from "react";

import type { ExploreAsset } from "@/lib/explore-contract";
import { releaseDecodedAudio } from "../../../components/WaveForm/waveformUtils";
import { useExploreRuntime } from "../../../runtime/ExploreRuntime";

export type AudioSourceUrl =
  | { status: "loading" }
  | { status: "ready"; url: string }
  | { status: "error" };

/**
 * Resolves a playable + fetchable URL for the trimmer. The waveform decodes the
 * bytes with `fetch`, which CSP blocks for `file:`, so Desktop hands back a
 * `blob:` URL instead of the raw path.
 */
export function useAudioSourceUrl(asset: ExploreAsset | null): AudioSourceUrl {
  const { audioSourceUrl } = useExploreRuntime();
  const [state, setState] = useState<AudioSourceUrl>({ status: "loading" });

  useEffect(() => {
    if (asset == null) {
      setState({ status: "loading" });
      return;
    }

    let cancelled = false;
    let revoke: (() => void) | null = null;

    setState({ status: "loading" });
    void audioSourceUrl(asset)
      .then((resolved) => {
        // Decoded PCM is cached per-src. A released URL can never be fetched
        // again, so evict it or every modal reopen leaks a full buffer.
        const release = () => {
          releaseDecodedAudio(resolved.url);
          resolved.release?.();
        };
        if (cancelled) {
          release();
          return;
        }
        revoke = release;
        setState({ status: "ready", url: resolved.url });
      })
      .catch(() => {
        if (!cancelled) setState({ status: "error" });
      });

    return () => {
      cancelled = true;
      revoke?.();
    };
  }, [asset, audioSourceUrl]);

  return state;
}
