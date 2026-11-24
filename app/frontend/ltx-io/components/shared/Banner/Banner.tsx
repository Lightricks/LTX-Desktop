import { clsx } from "clsx";

import { Text } from "@ds/Text/Text";

import styles from "./Banner.module.scss";

export const BANNER_TYPE_OPTIONS = ["standard", "monetisation"] as const;

type BannerType = (typeof BANNER_TYPE_OPTIONS)[number];

export type BannerProps = {
  type?: BannerType;
  text?: string;
  button?: React.ReactNode;
  className?: string;
};

export function Banner({
  type = "standard",
  text,
  button,
  className,
}: BannerProps) {
  return (
    <div
      className={clsx(
        styles.container,
        styles[type],
        className,
        "inverted-theme-style",
      )}
    >
      <Text variant="body" size="md">
        {text}
      </Text>
      {button}
    </div>
  );
}
