import { useDevFlags } from '../contexts/DevFlagsContext'

// Single source of truth for whether the "Custom IC-LoRA" UI is available.
// Off by default via the dev flag. Built-in Canny/Depth IC-LoRA is unaffected.
// The custom option only appears inside IC-LoRA mode, which is already gated by
// the existing API-only check — so this flag is all the extra gating it needs.
export function useCustomIcLoraEnabled(): boolean {
  return useDevFlags().flags.customIcLora
}
