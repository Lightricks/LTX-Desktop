import { ExampleEmptyMedia } from "./ExampleEmptyMedia";
import { packagedExploreExampleUrl } from "../../../assets/packaged-explore-assets";

export function ImageToVideoExampleEmpty() {
  return (
    <ExampleEmptyMedia
      posterUrl={packagedExploreExampleUrl("image-to-video-example-poster")}
      videoUrl={packagedExploreExampleUrl("image-to-video-example-video")}
    />
  );
}
