import React from "react";

import { Badge } from "@ds/Badge/Badge";

export type DefaultThumbnailRatio = "1:1" | "3:4";
export type GalleryButtonThumbnailRatio = "1:1" | "2:1";
export type ThumbnailRatio = DefaultThumbnailRatio | GalleryButtonThumbnailRatio;

export type ThumbnailBaseDescriptor = {
  id: string;
  title?: string;
  badge?: React.ReactElement<typeof Badge>;
  ratio?: DefaultThumbnailRatio;
};

type VideoThumbnailDescriptor = ThumbnailBaseDescriptor & {
  type: "video";
  imageUrl: string;
  videoUrl: string;
  autoplay: boolean;
};

type ImageThumbnailDescriptor = ThumbnailBaseDescriptor & {
  type: "image";
  imageUrl: string;
  isActive?: boolean;
  isStatic?: boolean;
};

type IconThumbnailDescriptor = ThumbnailBaseDescriptor & {
  type: "icon";
  icon?: React.ComponentType; // If no icon is passed, the default will be the "none" icon
};

export type GalleryButtonThumbnailDescriptor = Omit<ThumbnailBaseDescriptor, "ratio"> & {
  type: "galleryButton";
  icon: React.ReactNode;
  ratio?: GalleryButtonThumbnailRatio;
};

type PendingThumbnailDescriptor = ThumbnailBaseDescriptor & {
  type: "pending";
  progress?: number;
  imageUrl?: string;
};

type UnsupportedThumbnailDescriptor = ThumbnailBaseDescriptor & {
  type: "unsupported";
  imageUrl?: string;
  tooltipTitle?: string;
};

type RetryThumbnailDescriptor = ThumbnailBaseDescriptor & {
  type: "retry";
  onRetry?: () => void;
};

type CTAThumbnailDescriptor = ThumbnailBaseDescriptor & {
  type: "cta";
  message: string;
  ctaText: string;
  onCTA?: () => void;
};

export type SkeletonThumbnailDescriptor = Omit<ThumbnailBaseDescriptor, "title"> & {
  type: "skeleton";
  showTitle?: boolean;
};

// This is intentionally not part of the Thumbnail component, but is used in the ThumbnailsList component
export type CustomElementThumbnailDescriptor = ThumbnailBaseDescriptor & {
  type: "customElement";
  element: React.ReactNode;
};

export type ThumbnailDescriptor =
  | VideoThumbnailDescriptor
  | ImageThumbnailDescriptor
  | IconThumbnailDescriptor
  | GalleryButtonThumbnailDescriptor
  | UnsupportedThumbnailDescriptor
  | PendingThumbnailDescriptor
  | RetryThumbnailDescriptor
  | CTAThumbnailDescriptor
  | SkeletonThumbnailDescriptor;

export function hasImage(
  descriptor: ThumbnailDescriptor,
): descriptor is
  | ImageThumbnailDescriptor
  | VideoThumbnailDescriptor
  | PendingThumbnailDescriptor
  | UnsupportedThumbnailDescriptor {
  return (
    (descriptor.type === "image" ||
      descriptor.type === "video" ||
      descriptor.type === "pending" ||
      descriptor.type === "unsupported") &&
    descriptor.imageUrl !== ""
  );
}

export function hasTitle(
  descriptor: ThumbnailDescriptor | CustomElementThumbnailDescriptor,
): descriptor is (
  | VideoThumbnailDescriptor
  | ImageThumbnailDescriptor
  | IconThumbnailDescriptor
  | GalleryButtonThumbnailDescriptor
  | UnsupportedThumbnailDescriptor
  | PendingThumbnailDescriptor
  | RetryThumbnailDescriptor
  | CTAThumbnailDescriptor
) & { title: string } {
  return descriptor.type !== "skeleton" && descriptor.title !== undefined;
}
