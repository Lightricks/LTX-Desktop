import { useEffect, useState } from "react";

import type { ExploreListedAsset } from "@/lib/explore-contract";
import { AudioLightBoxPlayer } from "@/ltx-io/components/AudioPlayer/AudioLightBoxPlayer";
import { releaseDecodedAudio } from "@/ltx-io/components/WaveForm/waveformUtils";
import { useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";
import { useFeatureFormMuteStore } from "@/ltx-io/stores/featureFormMuteStore";

import styles from "./AssetsLightbox.module.scss";

export function AssetsLightboxAudio({
  asset,
}: {
  asset: ExploreListedAsset;
}) {
  const { waveformUrlForAsset } = useExploreRuntime();
  const isMuted = useFeatureFormMuteStore((state) => state.isMuted);
  const setMuted = useFeatureFormMuteStore((state) => state.setMuted);
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let createdUrl: string | null = null;

    void waveformUrlForAsset({
      id: asset.id,
      path: asset.path,
      bytes_url: asset.bytes_url,
    })
      .then((url) => {
        if (!url) {
          if (!cancelled) setFailed(true);
          return;
        }
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        createdUrl = url;
        setSrc(url);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });

    return () => {
      cancelled = true;
      setSrc(null);
      setFailed(false);
      if (createdUrl) {
        releaseDecodedAudio(createdUrl);
        URL.revokeObjectURL(createdUrl);
      }
    };
  }, [asset.bytes_url, asset.id, asset.path, waveformUrlForAsset]);

  if (failed) {
    return <div className={styles.brokenMedia}>Media unavailable</div>;
  }

  if (!src) {
    return <div className={styles.audio} aria-hidden />;
  }

  return (
    <div className={styles.audio}>
      <AudioLightBoxPlayer
        src={src}
        autoPlay
        muted={isMuted}
        onMute={() => setMuted(!isMuted)}
        onError={() => setFailed(true)}
      />
    </div>
  );
}
