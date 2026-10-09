import { useMemo } from "react";

import { useCreateTextToVideo } from "../../hooks/useCreateTextToVideo";
import { useVideoGenerationModelSpecs } from "../../hooks/useVideoGenerationModelSpecs";
import type { VideoFeatureContext } from "../../lib/videoFieldPolicy";
import { textToVideoDefinition } from "./definitions/textToVideo";
import { FeatureFormScreen } from "./FeatureFormScreen";
import { ExampleEmpty } from "./results/ExampleEmpty";
import { ResultFrame } from "./results/ResultFrame";
import { textToVideoFormPresentation } from "./TextToVideoFormPresentation";

export function TextToVideoFeatureScreen() {
  const specsQuery = useVideoGenerationModelSpecs();
  const createGeneration = useCreateTextToVideo();
  const context = useMemo<VideoFeatureContext>(
    () => ({ specs: specsQuery.data }),
    [specsQuery.data],
  );

  return (
    <FeatureFormScreen
      definition={textToVideoDefinition}
      context={context}
      contextReady={specsQuery.isSuccess}
      resourceError={specsQuery.error}
      createGeneration={createGeneration}
      ResultFrameAdapter={ResultFrame}
      EmptyStateAdapter={ExampleEmpty}
      presentation={textToVideoFormPresentation}
    />
  );
}
