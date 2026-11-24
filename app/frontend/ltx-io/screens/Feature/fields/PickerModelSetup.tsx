import { isH3Variant } from "@/lib/h3-models";
import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ModelSetupPanel, modelSetupLabels, type ModelSetupKind } from "@/components/settings/ModelSetupPanel";
import styles from "@/components/settings/ModelSetupPanel.module.scss";
import { Button } from "@ds/Button/Button";
import { generationQueryKeys } from "../../../hooks/generationQueryKeys";
import { useExploreRuntime } from "../../../runtime/ExploreRuntime";
import type { FieldOption } from "../types";

type Props<TValue extends string | number> = {
  kind: ModelSetupKind;
  option: FieldOption<TValue> | undefined;
  onUse: (value: TValue) => void;
  onClose: () => void;
};

// Refresh the same catalog that normalizes the form before committing its new model.
// Otherwise a newly downloaded native offering can be reset to the previous one.
export function PickerModelSetup<TValue extends string | number>({ kind, option, onUse, onClose }: Props<TValue>) {
  const { api, host } = useExploreRuntime();
  const queryClient = useQueryClient();
  const [verified, setVerified] = useState(false);
  const [refreshError, setRefreshError] = useState("");
  const used = useRef(false);

  const refresh = useCallback(async () => {
    setRefreshError("");
    try {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: generationQueryKeys.allSpecs }, { throwOnError: true }),
        queryClient.invalidateQueries(
          { queryKey: [host === "remote" ? "paired-amd-models" : kind.startsWith("ltx-") ? "ltx-versions" : isH3Variant(kind) ? "h3-models" : "comfy-models"] }, { throwOnError: true },
        ),
      ]);
      setVerified(true);
    } catch (error) {
      setRefreshError(error instanceof Error ? error.message : "Could not refresh model availability.");
    }
  }, [kind, host, queryClient]);

  useEffect(() => {
    if (!used.current && verified && option != null && !option.disabledReason) {
      used.current = true;
      onUse(option.value);
    }
  }, [verified, option, onUse]);

  const blocked = verified && (!option || option.disabledReason);
  return (
    <div className={styles.setupCopy}>
      {host === "desktop" ? <ModelSetupPanel kind={kind} api={api} resumeSelection onReady={refresh} onClose={onClose} /> : (
        <section aria-label={`Set up ${modelSetupLabels[kind]} on the host PC`}>
          <p>On the host PC, open LTX Desktop - AMD 395 MAX → Settings → Models and set up {modelSetupLabels[kind]}. Then refresh here to use it for this video.</p>
          <Button size="sm" label="Refresh and use" onClick={refresh} />
          <Button appearance="neutral" hierarchy="plain" size="sm" label="Close" onClick={onClose} />
        </section>
      )}
      {(refreshError || blocked) && (
        <div role="status">
          <p>{refreshError || option?.disabledReason || "This model is not currently available for this workflow."}</p>
          <Button appearance="neutral" hierarchy="secondary" size="sm" label="Refresh availability" onClick={refresh} />
        </div>
      )}
    </div>
  );
}
