import type { components } from "@/generated/backend-openapi";
import { useCreateFeatureGeneration } from "./useCreateFeatureGeneration";

type CreateAudioToVideoRequest = components["schemas"]["CreateAudioToVideoRequest"];

export function useCreateAudioToVideo() {
  return useCreateFeatureGeneration((api, body: CreateAudioToVideoRequest) =>
    api.createAudioToVideo(body),
  );
}
