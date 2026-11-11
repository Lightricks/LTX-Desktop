import { AlertTriangle } from "lucide-react";
import { useState } from "react";

import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";

import { Checkbox } from "../../components/Checkbox/Checkbox";
import { Modal } from "../../components/shared/Modal/Modal";
import { type ConfirmableTier, riskConfirmCopy } from "@/lib/budget-risk";

import styles from "./RiskyGenerateDialog.module.scss";

export function RiskyGenerateDialog({
  tier,
  message,
  allowDontAskAgain,
  onConfirm,
  onAdjust,
}: {
  tier: ConfirmableTier;
  message: string;
  /** False on Remote: the phone user never opted out on Desktop's behalf. */
  allowDontAskAgain: boolean;
  onConfirm: (choice: { dontAskAgain: boolean }) => void;
  onAdjust: () => void;
}) {
  const [dontAskAgain, setDontAskAgain] = useState(false);
  const copy = riskConfirmCopy(tier);
  return (
    <Modal onClose={onAdjust} variant="secondary">
      <div className={styles.body}>
        <AlertTriangle className={styles.icon} aria-hidden />
        <Text as="h2" variant="heading" size="sm">
          {copy.title}
        </Text>
        <Text as="p" variant="body" size="md">
          {message}
        </Text>
        <Text as="p" variant="body" size="md">
          {copy.body}
        </Text>
        {allowDontAskAgain ? (
          <Checkbox
            checked={dontAskAgain}
            onChange={(event) => setDontAskAgain(event.target.checked)}
          >
            <Text as="span" variant="body" size="md">
              Don't ask again
            </Text>
          </Checkbox>
        ) : null}
        <div className={styles.actions}>
          <Button
            appearance="neutral"
            hierarchy="secondary"
            label="Adjust settings"
            onClick={onAdjust}
          />
          <Button
            appearance="brand"
            hierarchy="primary"
            label="Generate anyway"
            onClick={() => onConfirm({ dontAskAgain })}
          />
        </div>
      </div>
    </Modal>
  );
}
