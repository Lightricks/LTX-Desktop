import { type ReactNode } from "react";
import { Radio, Search, Settings } from "lucide-react";
import { useLocation } from "react-router";
import { Button } from "@/ds/Button/Button";
import { Text } from "@/ds/Text/Text";
import { SidePanelMenu } from "@/ds/SidePanelMenu/SidePanelMenu";
import type {
  SidePanelMenuSection,
  SidePanelMenuSectionItem,
} from "@/ds/SidePanelMenu/SidePanelMenuSection";
import ProjectsIcon from "@/ds/assets/Icons/Projects.svg?react";
import AssetsIcon from "@/ds/assets/Icons/SectionVideos.svg?react";
import HomeIcon from "@/assets/home/icons/Home/Line.svg?react";
import {
  isHomeSidebarFooterItemActive,
  isHomeSidebarNavItemActive,
  isHomeSidebarRecentProjectActive,
  planHomeSidebarFooterItems,
  planHomeSidebarNavEntries,
  selectHomeSidebarFooterItem,
  selectHomeSidebarNavItem,
  selectHomeSidebarRecentProject,
  type HomeSidebarActions,
  type HomeSidebarItemPlan,
  type HomeSidebarNavItem,
  type HomeSidebarNavItemId,
} from "./home-sidebar-items";
import { homeFeatureIcon } from "./homeFeatureIcons";
import styles from "./DesktopHomeSidebar.module.scss";
import { useRecentSidebarProjects } from "./useRecentSidebarProjects";
import { useRecentSidebarTools } from "./useRecentSidebarTools";
function SidebarKeyboardShortcut({ keys }: { keys: string }) {
  return (
    <span className={styles.navShortcut} aria-hidden>
      <Text as="span" variant="label" size="sm" className={styles.navShortcutKey}>
        {keys}
      </Text>
    </span>
  );
}

function HomeNavButton({
  isActive,
  label,
  icon,
  onClick,
  className,
  rightIcon,
}: {
  isActive: boolean;
  label: string;
  icon: ReactNode;
  onClick: () => void;
  className?: string;
  rightIcon?: ReactNode;
}) {
  return (
    <Button
      hierarchy="plain"
      label={label}
      leftIcon={icon}
      rightIcon={rightIcon}
      isActive={isActive}
      aria-current={isActive ? "page" : undefined}
      aria-label={label}
      onClick={onClick}
      className={className ?? styles.navButton}
    />
  );
}

function iconForNavItem(id: HomeSidebarNavItemId): ReactNode {
  switch (id) {
    case "home":
      return <HomeIcon />;
    case "assets":
      return <AssetsIcon />;
    case "remote":
      return <Radio className="h-3.5 w-3.5" />;
    case "projects":
      return <ProjectsIcon />;
    default:
      return homeFeatureIcon(id);
  }
}

export function HomeSidebarMenu({
  plan,
  actions,
  onNavigateComplete,
}: {
  plan: HomeSidebarItemPlan;
  actions: HomeSidebarActions;
  onNavigateComplete?: () => void;
}) {
  const location = useLocation();
  const recentFeatureIds = useRecentSidebarTools();
  const recentProjects = useRecentSidebarProjects();
  const navEntries = planHomeSidebarNavEntries(
    plan,
    recentFeatureIds,
    recentProjects,
  );
  const footerItems = planHomeSidebarFooterItems(plan);

  const renderNavButton = (item: HomeSidebarNavItem): SidePanelMenuSectionItem => ({
    type: "custom",
    node: (
      <HomeNavButton
        isActive={isHomeSidebarNavItemActive(location.pathname, item)}
        label={item.label}
        icon={iconForNavItem(item.id)}
        onClick={() => {
          selectHomeSidebarNavItem(item, actions);
          onNavigateComplete?.();
        }}
      />
    ),
  });

  const exploreToolsItem: SidePanelMenuSectionItem = {
    type: "custom",
    node: (
      <HomeNavButton
        isActive={false}
        label="Explore Tools"
        icon={<Search className="h-3.5 w-3.5" aria-hidden strokeWidth={2} />}
        rightIcon={<SidebarKeyboardShortcut keys="/" />}
        onClick={() => actions.openQuickSearch?.()}
      />
    ),
  };

  const pageItems: SidePanelMenuSectionItem[] = [];
  const recentProjectItems: SidePanelMenuSectionItem[] = [];
  const recentItems: SidePanelMenuSectionItem[] = [];
  const featuredItems: SidePanelMenuSectionItem[] = [];
  let recentProjectsLabel: string | undefined;
  let recentLabel: string | undefined;
  let featuredLabel: string | undefined;
  type SidebarBucket = "pages" | "recent-projects" | "recent" | "featured";
  let bucket: SidebarBucket = "pages";
  for (const entry of navEntries) {
    if (entry.kind === "section") {
      if (entry.id === "recent-projects") {
        bucket = "recent-projects";
        recentProjectsLabel = entry.label;
      } else if (entry.id === "recent") {
        bucket = "recent";
        recentLabel = entry.label;
      } else {
        bucket = "featured";
        featuredLabel = entry.label;
      }
      continue;
    }
    if (entry.kind === "recent-project") {
      recentProjectItems.push({
        type: "custom",
        node: (
          <HomeNavButton
            isActive={isHomeSidebarRecentProjectActive(
              location.pathname,
              entry.path,
            )}
            label={entry.project.name}
            icon={<ProjectsIcon />}
            className={styles.navButton}
            onClick={() => {
              selectHomeSidebarRecentProject(entry.project.id, actions);
              onNavigateComplete?.();
            }}
          />
        ),
      });
      continue;
    }
    const item = renderNavButton(entry.item);
    if (bucket === "recent") recentItems.push(item);
    else if (bucket === "featured") featuredItems.push(item);
    else {
      pageItems.push(item);
      if (entry.item.id === "home") {
        pageItems.push(exploreToolsItem);
      }
    }
  }

  const sectionLabelProps = {
    labelVariant: "body" as const,
    labelSize: "md" as const,
    separatorClassName: styles.hideSeparator,
    labelClassName: styles.featuredLabel,
  };

  const sections: SidePanelMenuSection[] = [{ items: pageItems }];
  if (recentProjectItems.length > 0 && recentProjectsLabel !== undefined) {
    sections.push({
      label: recentProjectsLabel,
      items: recentProjectItems,
      ...sectionLabelProps,
    });
  }
  if (recentItems.length > 0 && recentLabel !== undefined) {
    sections.push({ label: recentLabel, items: recentItems, ...sectionLabelProps });
  }
  if (featuredItems.length > 0 && featuredLabel !== undefined) {
    sections.push({ label: featuredLabel, items: featuredItems, ...sectionLabelProps });
  }

  const footerSectionItems: SidePanelMenuSectionItem[] = [];
  for (const item of footerItems) {
    switch (item.id) {
      case "settings":
        footerSectionItems.push({
          type: "custom",
          node: (
            <HomeNavButton
              isActive={isHomeSidebarFooterItemActive(location.pathname, item, {
                settingsModalOpen: actions.isSettingsModalOpen,
              })}
              label={item.label}
              icon={<Settings className="h-3.5 w-3.5" />}
              onClick={() => {
                selectHomeSidebarFooterItem(item, actions);
                onNavigateComplete?.();
              }}
            />
          ),
        });
        break;
      default: {
        throw new Error(`Unhandled sidebar footer item: ${item.id}`);
      }
    }
  }

  return (
    <SidePanelMenu
      className={styles.flushFooter}
      sections={sections}
      footer={
        footerSectionItems.length > 0
          ? {
              items: footerSectionItems,
              separatorClassName: styles.footerSeparator,
            }
          : undefined
      }
    />
  );
}
