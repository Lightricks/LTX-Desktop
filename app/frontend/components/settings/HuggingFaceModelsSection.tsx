import { useHfAuth } from "../../hooks/use-hf-auth";
import { Button as DsButton } from "@/ds/Button/Button";
import { Text } from "@/ds/Text/Text";

import styles from "./SettingsContent.module.scss";

export function HuggingFaceModelsSection() {
  const { hfAuthStatus, hfAuthPolling, startHuggingFaceLogin, handleHuggingFaceLogout } =
    useHfAuth(true);

  return (
    <div className={styles.modelsManagementHfRow}>
      <div className={styles.modelsManagementHfCopy}>
        <Text as="h3" variant="heading" size="xs">
          Hugging Face
        </Text>
        {hfAuthStatus !== "authenticated" ? (
          <Text as="p" variant="body" size="sm" className={styles.apiKeysDescriptionLine}>
            Sign in to download gated models (such as LTX 2.5) and accept Hugging Face licenses.
          </Text>
        ) : null}
      </div>
      {hfAuthStatus === "authenticated" ? (
        <DsButton
          appearance="neutral"
          hierarchy="secondary"
          size="md"
          label="Sign out"
          onClick={handleHuggingFaceLogout}
          className={styles.modelsManagementHfAction}
        />
      ) : (
        <DsButton
          appearance="neutral"
          hierarchy="secondary"
          size="md"
          label={hfAuthPolling ? "Waiting for sign in…" : "Sign in with Hugging Face"}
          disabled={hfAuthPolling}
          onClick={() => void startHuggingFaceLogin()}
          className={styles.modelsManagementHfAction}
        />
      )}
    </div>
  );
}
