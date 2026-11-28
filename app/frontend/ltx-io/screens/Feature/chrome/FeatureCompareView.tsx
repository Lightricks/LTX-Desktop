import { VideoCompareCurtain } from "../../../components/VideoCompareCurtain/VideoCompareCurtain";
import { SourceResultToggle } from "../../../components/SourceResultToggle/SourceResultToggle";

import type { FeatureCompare } from "./featureCompare";

/** Fills its positioned parent. */
export function FeatureCompareView({
  compare,
  title,
}: {
  compare: FeatureCompare;
  title: string;
}) {
  if (compare.mode === "curtain") {
    return (
      <VideoCompareCurtain
        beforeUrl={compare.sourceVideoUrl}
        afterUrl={compare.result.videoUrl}
        afterPosterUrl={compare.result.posterUrl}
        beforeLabel="Before"
        afterLabel="After"
        ariaLabel={`${title} before and after`}
      />
    );
  }
  return (
    <SourceResultToggle
      key={compare.result.videoUrl}
      source={compare.source}
      result={compare.result}
      ariaLabel={title}
    />
  );
}
