import { useCallback } from "react";
import { useNavigate } from "react-router";

// eslint-disable-next-line no-restricted-imports
import { LtxioQuickSearch } from "@/ltx-io/components/QuickSearch/LtxioQuickSearch";
// eslint-disable-next-line no-restricted-imports
import type { QuickSearchNavigateOptions } from "@/ltx-io/components/QuickSearch/quickSearchNavigation";
import { useProjects } from "../../contexts/ProjectContext";

export function DesktopLtxioQuickSearch() {
  const navigate = useNavigate();
  const { clearActiveProject } = useProjects();
  const onNavigate = useCallback(
    (path: string, options: QuickSearchNavigateOptions) => {
      if (options.clearActiveProject) clearActiveProject();
      navigate(path);
    },
    [clearActiveProject, navigate],
  );
  return <LtxioQuickSearch onNavigate={onNavigate} />;
}
