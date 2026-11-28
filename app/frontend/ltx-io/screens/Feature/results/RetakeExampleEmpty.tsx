import { ExampleEmptyMedia } from "./ExampleEmptyMedia";
import { packagedExploreExampleUrl } from "../../../assets/packaged-explore-assets";

export function RetakeExampleEmpty() {
  return (
    <ExampleEmptyMedia
      posterUrl={packagedExploreExampleUrl("retake-example-poster")}
      videoUrl={packagedExploreExampleUrl("retake-example-video")}
    />
  );
}
