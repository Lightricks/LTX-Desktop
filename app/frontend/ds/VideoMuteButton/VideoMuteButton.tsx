import { clsx } from "clsx";
import { type MouseEvent } from "react";

import { Button } from "@ds/Button/Button";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import MuteIcon from "@ds/assets/Icons/Audio/Mute.svg?react";
import UnMuteIcon from "@ds/assets/Icons/Audio/On.svg?react";

import styles from "./VideoMuteButton.module.scss";

/**
 * Shared mute/unmute toggle overlaid on a video preview. One style everywhere:
 * the design-system overlay/secondary icon button with a tooltip (the retake
 * preview's treatment). Callers own positioning via a wrapping element.
 * Always applies `nodrag` so canvas nodes don't steal the click, and
 * `pointer-events: auto` so the control stays clickable inside
 * `pointer-events: none` overlays. `onToggle` fires after the click's
 * propagation is stopped, so it's safe to use inside a click-to-play preview.
 */
export function VideoMuteButton({
  isMuted,
  onToggle,
  className,
}: {
  isMuted: boolean;
  onToggle: () => void;
  className?: string;
}) {
  const handleClick = (event: MouseEvent<HTMLButtonElement>): void => {
    event.stopPropagation();
    onToggle();
  };

  return (
    <Tooltip content={isMuted ? "Unmute" : "Mute"}>
      <Button
        appearance="overlay"
        hierarchy="secondary"
        size="md"
        isIconOnly
        leftIcon={isMuted ? <MuteIcon /> : <UnMuteIcon />}
        aria-label={isMuted ? "Unmute video" : "Mute video"}
        className={clsx(styles.root, "nodrag", className)}
        onClick={handleClick}
      />
    </Tooltip>
  );
}
