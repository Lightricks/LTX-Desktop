import { Tooltip, type TooltipProps } from '@ds/Tooltip/Tooltip'

/**
 * Tooltip for controls inside the log dialog. The dialog is above the default
 * tooltip layer, so the tooltip is portaled to the body, where it sits above overlays.
 */
export function LogTooltip(props: TooltipProps) {
  return <Tooltip portalContainer={document.body} {...props} />
}
