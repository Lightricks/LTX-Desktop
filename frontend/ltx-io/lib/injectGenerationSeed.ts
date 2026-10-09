/** Attach the generation seed to queued-generation create bodies. */
export function injectGenerationSeed<
  T extends { params: Record<string, unknown> },
>(body: T, seed: number): T {
  return {
    ...body,
    params: {
      ...body.params,
      seed,
    },
  };
}
