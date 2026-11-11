import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { useCallback, useMemo } from "react";
import { useNavigate, useParams } from "react-router";

import { customLoraNavId } from "@/lib/custom-lora";
import { confirmCatalogModelDelete } from "@/lib/model-catalog";
import { paths } from "@/paths";

import { useImageAssetSize } from "../../hooks/useAsset";
import { useCreateCustomLora } from "../../hooks/useCreateCustomLora";
import {
  useCustomLoraImport,
  useCustomLoras,
  useRemoveCustomLora,
} from "../../hooks/useCustomLoras";
import { useVideoGenerationModelSpecs } from "../../hooks/useVideoGenerationModelSpecs";
import type { Generation } from "../../lib/resultsFeedModel";
import { useExploreRuntime } from "../../runtime/ExploreRuntime";
import { useFeatureFormStore } from "../../stores/featureFormStore";

import { LockedWorkflow, PreflightCard } from "./CatalogInstallPreflight";
import { useAddCustomLoraModal } from "./CustomLoraFeatureScreen";
import { customLoraFormPresentation } from "./CustomLoraFormPresentation";
import styles from "./CustomLoraRunScreen.module.scss";
import { FeatureFormScreen } from "./FeatureFormScreen";
import {
  type CustomLoraContext,
  type CustomLoraItem,
  createCustomLoraDefinition,
  generationRanCustomLora,
} from "./definitions/customLora";
import { ResultFrame } from "./results/ResultFrame";
import { ResultStatus } from "./results/ResultStatus";

/**
 * The page of one saved custom LoRA. It works like the page of a catalog LoRA: a prompt, the
 * video controls, a strength slider, and the runs of this LoRA. Before the first run, the right
 * side says where the result will show.
 * When the weights are gone (the user deleted them in Settings > Models), the form is locked and
 * the card asks for the weights again: from the saved link, or from a file.
 */
export function CustomLoraRunScreen() {
  const { customLoraId = "" } = useParams();
  const lorasQuery = useCustomLoras();
  const specsQuery = useVideoGenerationModelSpecs();
  const createGeneration = useCreateCustomLora();
  const definition = useMemo(() => createCustomLoraDefinition(), []);

  const lora = lorasQuery.data?.find((item) => item.id === customLoraId);
  const startFrameId = useFeatureFormStore(
    (state) => state.getValues(customLoraNavId(customLoraId))?.startFrame?.assetId ?? null,
  );
  const startFrameSize = useImageAssetSize(startFrameId);
  const context = useMemo<CustomLoraContext>(
    () => ({ specs: specsQuery.data, lora, startFrameSize }),
    [lora, specsQuery.data, startFrameSize],
  );
  const generationFilter = useCallback(
    (generation: Generation) => generationRanCustomLora(generation.spec, customLoraId),
    [customLoraId],
  );

  if (lorasQuery.isError) {
    return <Notice message={lorasQuery.error.message} />;
  }
  if (lorasQuery.isPending) return null;
  if (lora === undefined) {
    return <Notice message="This LoRA was not found." showAddLink />;
  }

  const form = (
    <FeatureFormScreen
      definition={definition}
      context={context}
      contextReady={specsQuery.isSuccess}
      resourceError={specsQuery.error}
      createGeneration={createGeneration}
      ResultFrameAdapter={ResultFrame}
      EmptyStateAdapter={CustomLoraResultsEmpty}
      presentation={customLoraFormPresentation}
      generateEnabled={specsQuery.budgetSettingsReady}
      generationFilter={generationFilter}
      storeKey={customLoraNavId(lora.id)}
      customLoraId={lora.id}
    />
  );

  if (lora.downloaded) return form;
  return (
    <LockedWorkflow overlay={<WeightsNeededCard lora={lora} />}>{form}</LockedWorkflow>
  );
}

function CustomLoraResultsEmpty() {
  return (
    <ResultStatus title="Generate to start" body="Generated result will appear here" />
  );
}

function AddAnotherLora() {
  const { customLoraImport } = useExploreRuntime();
  const openAdd = useAddCustomLoraModal();
  // A phone cannot add a LoRA, so the missing-LoRA notice stays a message.
  if (customLoraImport == null) return null;
  return (
    <button type="button" className={styles.link} onClick={openAdd}>
      Add a custom LoRA
    </button>
  );
}

function Notice({
  message,
  showAddLink = false,
}: {
  message: string;
  showAddLink?: boolean;
}) {
  return (
    <div className={styles.notice}>
      <Text as="p" variant="body" size="md">
        {message}
      </Text>
      {showAddLink ? <AddAnotherLora /> : null}
    </div>
  );
}

/**
 * Asks for the weights of a LoRA whose weights are gone. Desktop only: a phone cannot put
 * weights on the computer, so it shows where to do it.
 */
function WeightsNeededCard({ lora }: { lora: CustomLoraItem }) {
  const { api, customLoraImport } = useExploreRuntime();
  const navigate = useNavigate();
  const importer = useCustomLoraImport(api, () => undefined);
  const remove = useRemoveCustomLora();
  const progress = importer.state.status === "running" ? importer.state.progress : null;
  const running = progress !== null;

  if (customLoraImport == null) {
    return (
      <PreflightCard
        title="Weights needed"
        description={`The weights of "${lora.name}" are not on the computer. Add them in LTX Desktop.`}
      >
        <span />
      </PreflightCard>
    );
  }

  const description = lora.can_download
    ? `The weights of "${lora.name}" are not on this computer. Download them again, or choose the file.`
    : `The weights of "${lora.name}" are not on this computer. Choose the .safetensors file again.`;

  return (
    <PreflightCard title="Weights needed" description={description}>
      <div className={styles.actions}>
        {lora.can_download ? (
          <Button
            appearance="brand"
            hierarchy="primary"
            size="lg"
            label={running ? `Downloading… ${Math.round(progress ?? 0)}%` : "Download"}
            isLoading={running}
            disabled={running}
            onClick={() => void importer.restore(lora.id, {})}
          />
        ) : null}
        <Button
          appearance={lora.can_download ? "neutral" : "brand"}
          hierarchy={lora.can_download ? "secondary" : "primary"}
          size="lg"
          label="Choose file"
          disabled={running}
          onClick={() => {
            void customLoraImport.pickWeightsFile().then((picked) => {
              if (picked != null) void importer.restore(lora.id, { source_path: picked });
            });
          }}
        />
        <Button
          appearance="danger"
          hierarchy="secondary"
          size="lg"
          label="Remove LoRA"
          disabled={running || remove.isPending}
          onClick={() => {
            if (!confirmCatalogModelDelete(lora.name)) return;
            remove.mutate(lora.id, { onSuccess: () => void navigate(paths.home) });
          }}
        />
      </div>
      {importer.state.status === "error" ? (
        <Text as="p" variant="body" size="sm" className={styles.error}>
          {importer.state.message}
        </Text>
      ) : null}
      {remove.isError ? (
        <Text as="p" variant="body" size="sm" className={styles.error}>
          {remove.error.message}
        </Text>
      ) : null}
    </PreflightCard>
  );
}
