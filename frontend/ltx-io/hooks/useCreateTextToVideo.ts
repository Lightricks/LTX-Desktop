import type { components } from "@/generated/backend-openapi";
import { useCreateFeatureGeneration } from "./useCreateFeatureGeneration";

type CreateTextToVideoRequest = components["schemas"]["CreateTextToVideoRequest"];

export function useCreateTextToVideo() {
  return useCreateFeatureGeneration((api, body: CreateTextToVideoRequest) =>
    api.createTextToVideo(body),
  );
}
