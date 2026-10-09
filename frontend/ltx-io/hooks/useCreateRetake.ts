import type { components } from "@/generated/backend-openapi";
import { useCreateFeatureGeneration } from "./useCreateFeatureGeneration";

type CreateRetakeRequest = components["schemas"]["CreateRetakeRequest"];

export function useCreateRetake() {
  return useCreateFeatureGeneration((api, body: CreateRetakeRequest) =>
    api.createRetake(body),
  );
}
