import { ExampleEmptyMedia } from "./ExampleEmptyMedia";

export function LoraRecipeExampleEmpty({
  videoUrl,
  posterUrl,
}: {
  videoUrl: string;
  posterUrl: string;
}) {
  return <ExampleEmptyMedia posterUrl={posterUrl} videoUrl={videoUrl} />;
}
