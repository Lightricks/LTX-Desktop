import type { ReactNode } from "react";

import FrameRateIcon from "@ds/assets/Icons/Frame.svg?react";
import AspectRatioSquareIcon from "@ds/assets/Icons/Ratio/1x1.svg?react";
import AspectRatioLandscapeIcon from "@ds/assets/Icons/Ratio/16x9.svg?react";
import AspectRatioUltrawideIcon from "@ds/assets/Icons/Ratio/21x9.svg?react";
import AspectRatio3x2Icon from "@ds/assets/Icons/Ratio/3x2.svg?react";
import AspectRatio4x3Icon from "@ds/assets/Icons/Ratio/4x3.svg?react";
import AspectRatio4x5Icon from "@ds/assets/Icons/Ratio/4x5.svg?react";
import AspectRatioPortraitIcon from "@ds/assets/Icons/Ratio/9x16.svg?react";
import ResolutionLgIcon from "@ds/assets/Icons/Resolution/Lg.svg?react";
import ResolutionMdIcon from "@ds/assets/Icons/Resolution/Md.svg?react";
import ResolutionSmIcon from "@ds/assets/Icons/Resolution/Sm.svg?react";
import ResolutionXlIcon from "@ds/assets/Icons/Resolution/Xl.svg?react";
import DurationIcon from "@ds/assets/Icons/StopWatch.svg?react";

import LtxLogo from "@/assets/home/icons/LTX-logo.svg?react";

import type { FieldOption } from "../types";

const RESOLUTION_ICON_MAP: Partial<Record<string, ReactNode>> = {
  original: <ResolutionSmIcon aria-hidden />,
  "540p": <ResolutionSmIcon aria-hidden />,
  "720p": <ResolutionSmIcon aria-hidden />,
  "1080p": <ResolutionSmIcon aria-hidden />,
  "1440p": <ResolutionMdIcon aria-hidden />,
  "2048p": <ResolutionLgIcon aria-hidden />,
  "2160p": <ResolutionLgIcon aria-hidden />,
  "4320p": <ResolutionXlIcon aria-hidden />,
};

function getAspectRatioIcon(value: string): ReactNode {
  switch (value) {
    case "1:1":
      return <AspectRatioSquareIcon aria-hidden />;
    case "21:9":
      return <AspectRatioUltrawideIcon aria-hidden />;
    case "3:2":
      return <AspectRatio3x2Icon aria-hidden />;
    case "4:3":
      return <AspectRatio4x3Icon aria-hidden />;
    case "4:5":
      return <AspectRatio4x5Icon aria-hidden />;
    case "9:16":
      return <AspectRatioPortraitIcon aria-hidden />;
    default:
      return <AspectRatioLandscapeIcon aria-hidden />;
  }
}

/**
 * Icon for a video-settings option field (model / duration / aspect / resolution
 * / fps), shared by every video feature presentation (T2V and LoRA recipes) so a
 * new icon or resolution tier is edited in one place. Only reads `field.dataKey`,
 * so it is schema-agnostic across the video forms.
 */
export function getVideoOptionIcon(
  field: { dataKey: string },
  option: FieldOption,
): ReactNode | undefined {
  if (field.dataKey === "model") return <LtxLogo aria-hidden />;
  if (field.dataKey === "duration") return <DurationIcon aria-hidden />;
  if (field.dataKey === "aspectRatio") {
    return getAspectRatioIcon(String(option.value));
  }
  if (field.dataKey === "resolution") {
    return RESOLUTION_ICON_MAP[String(option.value)];
  }
  if (field.dataKey === "fps") return <FrameRateIcon aria-hidden />;
  return undefined;
}
