import { ExampleEmptyMedia } from "./ExampleEmptyMedia";
import { packagedExploreExampleUrl } from "../../../assets/packaged-explore-assets";

export function ExampleEmpty() {
  return (
    <ExampleEmptyMedia
      posterUrl={packagedExploreExampleUrl("text-to-video-example-poster")}
      videoUrl={packagedExploreExampleUrl("text-to-video-example-video")}
    />
  );
}
