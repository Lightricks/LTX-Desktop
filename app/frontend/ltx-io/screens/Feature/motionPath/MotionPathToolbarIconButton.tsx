import type { ReactNode } from "react";

import { Button } from "@ds/Button/Button";
import { Tooltip } from "@ds/Tooltip/Tooltip";

export function MotionPathToolbarIconButton({
  tooltip,
  icon,
  label,
  onClick,
  disabled,
  pressed,
}: {
  tooltip: ReactNode;
  icon: ReactNode;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  pressed?: boolean;
}) {
  return (
    <Tooltip content={tooltip}>
      <div>
        <Button
          appearance="neutral"
          hierarchy="plain"
          size="md"
          isIconOnly
          leftIcon={icon}
          aria-label={label}
          aria-pressed={pressed}
          disabled={disabled}
          onClick={onClick}
        />
      </div>
    </Tooltip>
  );
}
