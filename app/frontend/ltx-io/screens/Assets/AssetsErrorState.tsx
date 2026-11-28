import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { ResultStatus } from "../Feature/results/ResultStatus";
import { ASSETS_ERROR_CTA, ASSETS_ERROR_DESCRIPTION, ASSETS_ERROR_TITLE } from "./assetsCopy";
import styles from "./AssetsLibraryScreen.module.scss";

export function AssetsErrorState({
  onRetry,
  variant = "page",
}: {
  onRetry: () => void;
  variant?: "page" | "inline";
}) {
  const action = (
    <Button
      appearance="brand"
      hierarchy="primary"
      size="md"
      label={ASSETS_ERROR_CTA}
      onClick={onRetry}
    />
  );

  switch (variant) {
    case "inline":
      return (
        <div className={styles.inlineError} role="alert">
          <div className={styles.inlineErrorCopy}>
            <Text as="span" variant="heading" size="sm">
              {ASSETS_ERROR_TITLE}
            </Text>
            <Text as="p" variant="body" size="md">
              {ASSETS_ERROR_DESCRIPTION}
            </Text>
          </div>
          {action}
        </div>
      );
    case "page":
      return (
        <ResultStatus
          title={ASSETS_ERROR_TITLE}
          body={ASSETS_ERROR_DESCRIPTION}
          action={action}
        />
      );
    default: {
      const _exhaustive: never = variant;
      return _exhaustive;
    }
  }
}
