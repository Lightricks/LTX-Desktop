import { Button } from "@ds/Button/Button";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import type { ReactNode } from "react";

/** An icon button over a media preview, with a tooltip. */
export function OverlayButton({
  label,
  ariaLabel,
  icon,
  onClick,
}: {
  label: string;
  ariaLabel: string;
  icon: ReactNode;
  onClick: () => void;
}) {
  return (
    <Tooltip content={label}>
      <span>
        <Button
          appearance="overlay"
          hierarchy="secondary"
          size="md"
          isIconOnly
          leftIcon={icon}
          aria-label={ariaLabel}
          onClick={(event) => {
            event.stopPropagation();
            onClick();
          }}
        />
      </span>
    </Tooltip>
  );
}
