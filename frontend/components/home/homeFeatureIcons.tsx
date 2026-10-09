import type { ComponentType, ReactNode, SVGProps } from "react";

import AdjustIcon from "@/ds/assets/Icons/Adjust.svg?react";
import AlphaGenIcon from "@/ds/assets/Icons/AlphaGen.svg?react";
import AudioIcon from "@/ds/assets/Icons/Audio.svg?react";
import BackdropIcon from "@/ds/assets/Icons/Backdrop.svg?react";
import CubeSketchIcon from "@/ds/assets/Icons/CubeSketch.svg?react";
import ExtendIcon from "@/ds/assets/Icons/ExtendDuration.svg?react";
import MagicWandIcon from "@/ds/assets/Icons/MagicWand.svg?react";
import PhotoIcon from "@/ds/assets/Icons/Photo.svg?react";
import PickerIcon from "@/ds/assets/Icons/Picker.svg?react";
import VideoCameraIcon from "@/ds/assets/Icons/VideoCamera.svg?react";
import type { HomeFeatureId } from "@/lib/home-features";

type SvgIcon = ComponentType<SVGProps<SVGSVGElement>>;

/**
 * Home feature icons that differ from the default. They must match the LTX Studio sidebar
 * (`FEATURED_ICONS` in `ltxioSidebarFeaturedIcons.ts` in the ltx-studio repo). Copy the
 * SVG from `infinity/src/assets/Icons/` in ltx-studio, unchanged, into
 * `frontend/ds/assets/Icons/`. A feature that ltx.io gives no icon uses the video
 * camera, as ltx.io does, so such a feature needs no entry.
 */
const FEATURE_ICONS: Partial<Record<HomeFeatureId, SvgIcon>> = {
  "image-to-video": PhotoIcon,
  "audio-to-video": AudioIcon,
  extend: ExtendIcon,
  "alpha-gen": AlphaGenIcon,
  deblur: MagicWandIcon,
  colorization: PickerIcon,
  "clean-plate": BackdropIcon,
  decompression: AdjustIcon,
  "layout-to-render": CubeSketchIcon,
};

export function homeFeatureIcon(id: HomeFeatureId): ReactNode {
  const Icon = FEATURE_ICONS[id] ?? VideoCameraIcon;
  return <Icon aria-hidden />;
}
