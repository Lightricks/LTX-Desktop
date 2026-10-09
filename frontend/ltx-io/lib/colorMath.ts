export type Hsv = { h: number; s: number; v: number };

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));

/** Hue in degrees [0, 360). Saturation and value in [0, 1]. Input is `#rrggbb`. */
export function hexToHsv(hex: string): Hsv {
  const red = Number.parseInt(hex.slice(1, 3), 16) / 255;
  const green = Number.parseInt(hex.slice(3, 5), 16) / 255;
  const blue = Number.parseInt(hex.slice(5, 7), 16) / 255;
  const max = Math.max(red, green, blue);
  const delta = max - Math.min(red, green, blue);
  let hue = 0;
  if (delta > 0) {
    if (max === red) hue = ((green - blue) / delta) % 6;
    else if (max === green) hue = (blue - red) / delta + 2;
    else hue = (red - green) / delta + 4;
    hue *= 60;
    if (hue < 0) hue += 360;
  }
  return { h: hue, s: max === 0 ? 0 : delta / max, v: max };
}

export function hsvToHex({ h, s, v }: Hsv): string {
  const hue = ((h % 360) + 360) % 360;
  const saturation = clamp01(s);
  const value = clamp01(v);
  const channel = (offset: number): number => {
    const k = (offset + hue / 60) % 6;
    return value - value * saturation * Math.max(0, Math.min(k, 4 - k, 1));
  };
  const toHex = (part: number): string =>
    Math.round(part * 255)
      .toString(16)
      .padStart(2, "0");
  return `#${toHex(channel(5))}${toHex(channel(3))}${toHex(channel(1))}`;
}
