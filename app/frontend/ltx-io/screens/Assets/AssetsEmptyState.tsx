import { Button } from "@ds/Button/Button";
import { ResultStatus } from "../Feature/results/ResultStatus";
import {
  ASSETS_EMPTY_CTA,
  ASSETS_EMPTY_DESCRIPTION,
  ASSETS_EMPTY_TITLE,
  ASSETS_FILTERED_EMPTY_CTA,
  ASSETS_FILTERED_EMPTY_TITLE,
} from "./assetsCopy";

export function AssetsEmptyState({
  filtered,
  onGoHome,
  onClearFilters,
}: {
  filtered: boolean;
  onGoHome: () => void;
  onClearFilters: () => void;
}) {
  if (filtered) {
    return (
      <ResultStatus
        title={ASSETS_FILTERED_EMPTY_TITLE}
        action={
          <Button
            appearance="brand"
            hierarchy="primary"
            size="md"
            label={ASSETS_FILTERED_EMPTY_CTA}
            onClick={onClearFilters}
          />
        }
      />
    );
  }

  return (
    <ResultStatus
      title={ASSETS_EMPTY_TITLE}
      body={ASSETS_EMPTY_DESCRIPTION}
      action={
        <Button
          appearance="brand"
          hierarchy="primary"
          size="md"
          label={ASSETS_EMPTY_CTA}
          onClick={onGoHome}
        />
      }
    />
  );
}
