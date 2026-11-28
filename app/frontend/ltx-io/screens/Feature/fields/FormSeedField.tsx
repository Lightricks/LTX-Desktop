import { Text } from "@ds/Text/Text";

import { SeedField } from "@/components/SeedField";
import { useGenerationSeed } from "@/hooks/use-generation-seed";

import { useExploreRuntime } from "../../../runtime/ExploreRuntime";
import formStyles from "../FeatureFormView.module.scss";

/** The Seed row on Generate forms. Shares its value with Settings and the remote app. */
export function FormSeedField() {
  const { api } = useExploreRuntime();
  const { seed, locked, ready, setSeed, setLocked } = useGenerationSeed(api);

  return (
    <div className={formStyles.field}>
      <Text as="span" variant="body" size="md" className={formStyles.fieldLabel}>
        Seed
      </Text>
      <SeedField
        value={seed}
        onChange={setSeed}
        ariaLabel="Seed"
        locked={locked}
        onLockedChange={setLocked}
        disabled={!ready}
      />
    </div>
  );
}
