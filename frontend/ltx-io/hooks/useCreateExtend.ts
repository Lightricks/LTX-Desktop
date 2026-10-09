import type { components } from "@/generated/backend-openapi";
import { useCreateFeatureGeneration } from "./useCreateFeatureGeneration";

type CreateExtendRequest = components["schemas"]["CreateExtendRequest"];

export function useCreateExtend() {
  return useCreateFeatureGeneration((api, body: CreateExtendRequest) =>
    api.createExtend(body),
  );
}
