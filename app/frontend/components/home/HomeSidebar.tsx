import type {
  HomeSidebarActions,
  HomeSidebarItemPlan,
} from "./home-sidebar-items";
import { HomeSidebarDrawer } from "./HomeSidebarDrawer";
import { HomeSidebarRail } from "./HomeSidebarRail";
import { useHomeSidebarCompact } from "./useHomeSidebarCompact";

export function HomeSidebar({
  plan,
  actions,
  portalContainer,
}: {
  plan: HomeSidebarItemPlan;
  actions: HomeSidebarActions;
  portalContainer?: HTMLElement | null;
}) {
  const compact = useHomeSidebarCompact();
  if (compact) {
    return (
      <HomeSidebarDrawer plan={plan} actions={actions} />
    );
  }
  return (
    <HomeSidebarRail
      plan={plan}
      actions={actions}
      portalContainer={portalContainer}
    />
  );
}
