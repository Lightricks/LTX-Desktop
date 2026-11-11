import { ExampleEmptyMedia } from "./ExampleEmptyMedia";
import { packagedExploreExampleUrl } from "../../../assets/packaged-explore-assets";

export function KeyframesExampleEmpty() {
  return (
    <ExampleEmptyMedia
      posterUrl={packagedExploreExampleUrl("keyframes-example-poster")}
      videoUrl={packagedExploreExampleUrl("keyframes-example-video")}
    />
  );
}
