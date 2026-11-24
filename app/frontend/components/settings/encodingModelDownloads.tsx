import { AlertCircle } from "lucide-react";

import { Text } from "@/ds/Text/Text";

import contentStyles from "./SettingsContent.module.scss";

export function SettingsHintWarning({ children }: { children: React.ReactNode }) {
  return (
    <Text as="p" variant="body" size="xs" className={contentStyles.hintWarning}>
      <span className="inline-flex items-start gap-1.5">
        <AlertCircle className="h-3 w-3 flex-shrink-0 mt-0.5" aria-hidden />
        <span>{children}</span>
      </span>
    </Text>
  );
}
