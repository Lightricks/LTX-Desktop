/** Consume Escape before the surrounding assistant can interpret it as Stop. */
export function consumeQueueEscape(
  event: Pick<Event, "preventDefault" | "stopPropagation">,
  dragging: boolean,
  closePanel: () => void,
): void {
  event.preventDefault();
  // The drag sensor must still receive Escape, but the assistant checks defaultPrevented.
  if (dragging) return;
  event.stopPropagation();
  closePanel();
}
