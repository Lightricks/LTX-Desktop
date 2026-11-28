import { ExampleEmptyMedia } from "./ExampleEmptyMedia";
import { packagedExploreExampleUrl } from "../../../assets/packaged-explore-assets";

export function ExtendExampleEmpty() {
  return (
    <ExampleEmptyMedia
      posterUrl={packagedExploreExampleUrl("extend-example-poster")}
      videoUrl={packagedExploreExampleUrl("extend-example-video")}
    />
  );
}
