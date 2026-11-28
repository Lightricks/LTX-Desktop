import type { components } from "@/generated/backend-openapi";
import { useCreateFeatureGeneration } from "./useCreateFeatureGeneration";

type CreateImageToVideoRequest = components["schemas"]["CreateImageToVideoRequest"];

export function useCreateImageToVideo() {
  return useCreateFeatureGeneration((api, body: CreateImageToVideoRequest) =>
    api.createImageToVideo(body),
  );
}
