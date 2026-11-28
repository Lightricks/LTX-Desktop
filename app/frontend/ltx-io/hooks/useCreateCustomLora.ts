import type { components } from "@/generated/backend-openapi";
import { useCreateFeatureGeneration } from "./useCreateFeatureGeneration";

type CreateCustomLoraRequest = components["schemas"]["CreateCustomLoraRequest"];

export function useCreateCustomLora() {
  return useCreateFeatureGeneration((api, body: CreateCustomLoraRequest) =>
    api.createCustomLora(body),
  );
}
