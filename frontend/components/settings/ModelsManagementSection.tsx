import { HuggingFaceModelsSection } from "./HuggingFaceModelsSection";
import { ModelsFolderSection } from "./ModelsFolderSection";
import styles from "./SettingsContent.module.scss";

/** Hugging Face sign-in and on-disk models folder — one surfaced block, no border. */
export function ModelsManagementSection() {
  return (
    <div className={styles.modelsManagementPanel}>
      <HuggingFaceModelsSection />
      <ModelsFolderSection nested />
    </div>
  );
}
