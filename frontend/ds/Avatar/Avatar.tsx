import { clsx } from "clsx";
import { forwardRef } from "react";

import styles from "./Avatar.module.scss";

export const HIGHLIGHT_COLORS = [
  "green",
  "orange",
  "purple",
  "teal",
  "pink",
  "lime",
  "indigo",
  "yellow",
  "magenta",
  "red",
  "blue",
  "primary",
] as const;

export type HighlightColor = (typeof HIGHLIGHT_COLORS)[number];

export type AvatarProps = {
  userName?: string | null;
  userEmail: string;
  imageUrl?: string | null;
  onClick?: (event: React.MouseEvent) => void;
  appearance?: "square" | "circle";
  size?: "sm" | "md" | "lg" | "xl";
  state?: "enabled" | "selected" | "disabled" | "active" | "readonly";
  className?: string;
  shouldHighlightBorder?: boolean;
  highlightColor?: HighlightColor;
};

export const Avatar = forwardRef<HTMLDivElement | HTMLButtonElement, AvatarProps>(
  function Avatar(
    {
      userName,
      userEmail,
      imageUrl,
      onClick,
      appearance = "square",
      size = "md",
      state = "enabled",
      className,
      shouldHighlightBorder = false,
      highlightColor,
      ...restProps
    },
    ref,
  ) {
    const initials = getUserInitials(userEmail, userName);
    const hasContent = !!imageUrl || !!initials;

    const sharedClassName = clsx(
      styles.avatar,
      styles[appearance],
      styles[size],
      styles[state],
      highlightColor && hasContent && styles.backgroundColor,
      shouldHighlightBorder && styles.userBorderColor,
      !hasContent && styles.fallback,
      className,
    );

    const content = imageUrl ? (
      <img className={styles.image} src={imageUrl} alt={userEmail} />
    ) : (
      initials
    );

    const highlightProps =
      highlightColor && hasContent
        ? { "data-highlight-color": highlightColor }
        : undefined;

    if (onClick) {
      return (
        <button
          type="button"
          ref={ref as React.Ref<HTMLButtonElement>}
          className={sharedClassName}
          data-testid="avatar"
          {...highlightProps}
          onClick={onClick}
          {...restProps}
        >
          {content}
        </button>
      );
    }

    return (
      <div
        ref={ref as React.Ref<HTMLDivElement>}
        className={sharedClassName}
        data-testid="avatar"
        {...highlightProps}
        {...restProps}
      >
        {content}
      </div>
    );
  },
);

Avatar.displayName = "Avatar";

function getUserInitials(
  userEmail: string,
  userName?: string | null,
): string | undefined {
  let nameInitials = userName ? getNameInitials(userName) : undefined;
  if (!nameInitials) {
    nameInitials = userEmail.charAt(0).toUpperCase();
  }
  return nameInitials;
}

function getNameInitials(name: string): string {
  const pattern = /\b\w/g;
  return (name.match(pattern) ?? []).join("").substring(0, 2).toUpperCase();
}
