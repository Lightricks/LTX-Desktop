import { Fragment } from "react";
import { Link } from "react-router";

import { Button } from "@ds/Button/Button";
import InfoLineIcon from "@ds/assets/Icons/Info/Line.svg?react";

import { clsx } from "clsx";

import type { FeatureCrumb } from "./featureChromeModel";
import styles from "./FeatureBreadcrumbs.module.scss";

export function FeatureBreadcrumbs({
  trail,
  detailsLabel,
  onOpenDetails,
  variant = "default",
}: {
  trail: readonly FeatureCrumb[];
  detailsLabel: string;
  onOpenDetails: () => void;
  variant?: "default" | "topBar";
}) {
  if (trail.length === 0) {
    return null;
  }

  return (
    <nav
      className={clsx(
        styles.crumbs,
        variant === "topBar" && styles.crumbsInTopBar,
      )}
      aria-label="Breadcrumb"
    >
      {trail.map((crumb, index) => {
        const isLast = index === trail.length - 1;

        return (
          <Fragment key={crumb.id}>
            {index > 0 ? (
              <span className={styles.separator}> / </span>
            ) : null}
            {isLast ? (
              <span className={styles.current} aria-current="page">
                {crumb.label}
              </span>
            ) : crumb.to != null ? (
              <Link to={crumb.to} className={styles.crumbLink}>
                {crumb.label}
              </Link>
            ) : (
              <span className={styles.crumbText}>{crumb.label}</span>
            )}
          </Fragment>
        );
      })}
      <Button
        appearance="neutral"
        hierarchy="plain"
        size="md"
        isIconOnly
        leftIcon={<InfoLineIcon />}
        aria-label={detailsLabel}
        className={styles.infoButton}
        onClick={onOpenDetails}
      />
    </nav>
  );
}
