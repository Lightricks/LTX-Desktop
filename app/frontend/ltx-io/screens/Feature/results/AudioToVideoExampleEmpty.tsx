import { ExampleEmptyMedia } from "./ExampleEmptyMedia";
import { packagedExploreExampleUrl } from "../../../assets/packaged-explore-assets";

export function AudioToVideoExampleEmpty() {
  return (
    <ExampleEmptyMedia
      posterUrl={packagedExploreExampleUrl("audio-to-video-example-poster")}
      videoUrl={packagedExploreExampleUrl("audio-to-video-example-video")}
    />
  );
}
