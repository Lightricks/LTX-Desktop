import { useNavigate } from "react-router";
import { Button } from "@/ds/Button/Button";
import { ItemAction } from "@/ds/DropdownItems/ItemAction/ItemAction";
import { DropdownMenu } from "@/ds/DropdownMenu/DropdownMenu";
import HelpIcon from "@/assets/home/icons/Help.svg?react";
import { getHomeShellEntry } from "../../lib/home-shell";
import { paths } from "../../paths";
import { useAppChromeControls } from "../AppShell";
import fabStyles from "./DesktopHomeSupportFab.module.scss";
import styles from "./DesktopHomeSidebar.module.scss";
import { HOME_EXTERNAL_URLS, openHomeExternalUrl } from "./home-external-links";

export function DesktopHomeSupportMenu({
  portalContainer,
  onMenuStateChange,
  variant = "sidebar",
}: {
  portalContainer?: HTMLElement | null;
  onMenuStateChange?: (isOpen: boolean) => void;
  variant?: "sidebar" | "fab";
}) {
  const { openLogs } = useAppChromeControls();
  const navigate = useNavigate();

  const trigger =
    variant === "fab" ? (
      <Button
        appearance="neutral"
        hierarchy="elevated"
        size="md"
        isIconOnly
        aria-label="Support"
        leftIcon={<HelpIcon className={fabStyles.fabIcon} aria-hidden />}
        className={fabStyles.fab}
      />
    ) : (
      <Button
        hierarchy="plain"
        label="Support"
        leftIcon={<HelpIcon />}
        aria-label="Support"
        className={styles.navButton}
      />
    );

  return (
    <DropdownMenu
      side="top"
      align={variant === "fab" ? "end" : "start"}
      portalContainer={portalContainer}
      onMenuStateChange={onMenuStateChange}
      items={[
        {
          render: (
            <ItemAction
              text="Help center"
              onClick={() => openHomeExternalUrl(HOME_EXTERNAL_URLS.support)}
            />
          ),
        },
        {
          render: (
            <ItemAction
              text="Report an issue"
              onClick={() => openHomeExternalUrl(HOME_EXTERNAL_URLS.reportIssue)}
            />
          ),
        },
        {
          render: (
            <ItemAction
              text={getHomeShellEntry("dashboard").label}
              onClick={() => navigate(paths.dashboard)}
            />
          ),
        },
        {
          render: (
            <ItemAction text="Diagnostic logs" onClick={() => openLogs()} />
          ),
        },
      ]}
    >
      {trigger}
    </DropdownMenu>
  );
}
