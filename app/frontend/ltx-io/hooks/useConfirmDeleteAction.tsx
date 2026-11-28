import { useCallback, useState } from "react";

import { Button } from "@ds/Button/Button";
import { Flex } from "@ds/layout/Flex/Flex";
import { Text } from "@ds/Text/Text";

import {
  getSkipDeleteConfirmation,
  setSkipDeleteConfirmation,
} from "@/lib/delete-confirmation-preference";
import { Checkbox } from "@/ltx-io/components/Checkbox/Checkbox";
import { useModalContext } from "@/ltx-io/components/shared/Modal/ModalContext";

const DONT_ASK_AGAIN_LABEL = "Don't ask me again";

export type DeleteConfirmationOptions = {
  confirmationName: string;
  title: string;
  description: string;
  confirmLabel: string;
};

export function DeleteConfirmationDialog({
  title,
  description,
  confirmLabel,
  onDismiss,
  onConfirm,
}: {
  title: string;
  description: string;
  confirmLabel: string;
  onDismiss: () => void;
  onConfirm: () => void;
}) {
  const [dontAskAgain, setDontAskAgain] = useState(false);

  return (
    <Flex direction="column" gap="md" style={{ maxWidth: 400 }}>
      <Flex direction="column" gap="xs">
        <Text as="h2" variant="heading" size="sm">
          {title}
        </Text>
        <Text as="p" variant="body" size="md">
          {description}
        </Text>
      </Flex>
      <Checkbox
        checked={dontAskAgain}
        appearance="neutral"
        onChange={(event) => setDontAskAgain(event.currentTarget.checked)}
      >
        <Text as="span" variant="body" size="md">
          {DONT_ASK_AGAIN_LABEL}
        </Text>
      </Checkbox>
      <Flex gap="sm" justify="end">
        <Button
          data-autofocus
          appearance="neutral"
          hierarchy="plain"
          size="md"
          label="Cancel"
          onClick={onDismiss}
        />
        <Button
          appearance="danger"
          hierarchy="primary"
          size="md"
          label={confirmLabel}
          onClick={() => {
            if (dontAskAgain) {
              setSkipDeleteConfirmation(true);
            }
            onConfirm();
          }}
        />
      </Flex>
    </Flex>
  );
}

export function useConfirmDeleteAction() {
  const { hideConfirmation, showConfirmation } = useModalContext();

  return useCallback(
    (options: DeleteConfirmationOptions, onConfirm: () => void) => {
      if (getSkipDeleteConfirmation()) {
        onConfirm();
        return;
      }
      showConfirmation({
        confirmationName: options.confirmationName,
        content: (
          <DeleteConfirmationDialog
            title={options.title}
            description={options.description}
            confirmLabel={options.confirmLabel}
            onDismiss={() => hideConfirmation("cancel")}
            onConfirm={() => {
              hideConfirmation("confirm");
              onConfirm();
            }}
          />
        ),
      });
    },
    [hideConfirmation, showConfirmation],
  );
}
