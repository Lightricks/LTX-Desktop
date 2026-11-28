export const IC_LORA_CONTROL_TYPES = ["pose", "depth", "canny"] as const;

export type IcLoraControlType = (typeof IC_LORA_CONTROL_TYPES)[number];

/** The ltx.io control mode labels and captions. A new control type must add its entry here. */
export const IC_LORA_CONTROL_COPY: Record<IcLoraControlType, { label: string; caption: string }> = {
  pose: { label: "Pose", caption: "Use Pose as a video reference" },
  depth: { label: "Depth", caption: "Use the depth map as a video reference" },
  canny: { label: "Canny edges", caption: "Use the structural outlines of your original video" },
};

export function isIcLoraControlType(value: unknown): value is IcLoraControlType {
  return (IC_LORA_CONTROL_TYPES as readonly unknown[]).includes(value);
}
