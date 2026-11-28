import type { components } from "@/generated/backend-openapi";
import { useCreateFeatureGeneration } from "./useCreateFeatureGeneration";

type CreateKeyframesRequest = components["schemas"]["CreateKeyframesRequest"];

export function useCreateKeyframes() {
  return useCreateFeatureGeneration((api, body: CreateKeyframesRequest) =>
    api.createKeyframes(body),
  );
}
