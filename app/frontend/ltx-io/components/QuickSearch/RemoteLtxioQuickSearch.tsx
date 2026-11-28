import { useCallback } from "react";
import { useNavigate } from "react-router";

import { LtxioQuickSearch } from "./LtxioQuickSearch.tsx";
import type { QuickSearchNavigateOptions } from "./quickSearchNavigation.ts";

export function RemoteLtxioQuickSearch() {
  const navigate = useNavigate();
  const onNavigate = useCallback(
    (path: string, _options: QuickSearchNavigateOptions) => {
      navigate(path);
    },
    [navigate],
  );
  return <LtxioQuickSearch onNavigate={onNavigate} />;
}
