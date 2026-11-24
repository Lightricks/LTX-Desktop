import type { CreateTextToImageRequest } from "../components/ImageGenerator/imageGeneratorRequest";
import { useCreateFeatureGeneration } from "./useCreateFeatureGeneration";

export function useCreateTextToImage() {
  return useCreateFeatureGeneration((api, body: CreateTextToImageRequest) =>
    api.createTextToImage(body),
  );
}
