import { useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import { useQuery } from "@tanstack/react-query";

import { Button } from "@/ds/Button/Button";
import { Text } from "@/ds/Text/Text";

import { ResultStatus } from "../../components/shared/ResultStatus/ResultStatus.tsx";
import { generationQueryKeys } from "../../hooks/generationQueryKeys.ts";
import { unwrapApiResult } from "../../lib/unwrapApiResult.ts";
import { useExploreRuntime } from "../../runtime/ExploreRuntime.tsx";

import { DashboardPanels } from "./DashboardPanels.tsx";
import { dashboardLayout } from "./dashboardLayout.ts";
import { useDashboardSelection, type DashboardRange } from "./useDashboardSelection.ts";
import { useRefreshDashboardOnFinish } from "./useRefreshDashboardOnFinish.ts";
import styles from "./DashboardScreen.module.scss";

const RANGES: { id: DashboardRange; label: string }[] = [
  { id: "7d", label: "7 days" },
  { id: "30d", label: "30 days" },
  { id: "all", label: "All" },
];

export function DashboardScreen() {
  const { api } = useExploreRuntime();
  const tz = useMemo(
    () => Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC",
    [],
  );
  const { selection, ready, setSelection } = useDashboardSelection();
  useRefreshDashboardOnFinish();
  const range = selection.range;
  const pageRef = useRef<HTMLDivElement>(null);
  const width = useDashboardWidth(pageRef);
  const query = useQuery({
    queryKey: generationQueryKeys.dashboard(range, tz),
    enabled: ready,
    queryFn: async () => {
      let result = await api.getDashboard({ range, tz });
      if (!result.ok && result.error.code === "INVALID_TIMEZONE" && tz !== "UTC") {
        result = await api.getDashboard({ range, tz: "UTC" });
      }
      return unwrapApiResult(result);
    },
  });

  return (
    <div ref={pageRef} className={styles.page}>
      <header className={styles.header}>
        <div className={styles.heading}>
          <Text as="h1" variant="heading" size="lg" className={styles.title}>
            Activity Dashboard
          </Text>
          <Text as="p" variant="body" size="md" className={styles.muted}>
            Gen Space generations are not included.
          </Text>
        </div>
        <div className={styles.controls}>
          {RANGES.map((item) => (
            <button
              key={item.id}
              type="button"
              className={styles.control}
              aria-pressed={range === item.id}
              onClick={() => setSelection({ range: item.id })}
            >
              <Text as="span" variant="label" size="md">
                {item.label}
              </Text>
            </button>
          ))}
        </div>
      </header>
      {query.isPending ? (
        <Text as="p" variant="body" size="md" className={styles.muted}>
          Loading your generations…
        </Text>
      ) : null}
      {query.isError ? (
        <section className={styles.empty} role="alert">
          <ResultStatus
            title="Activity Dashboard isn't available right now"
            body="Your generations are still saved. Try loading them again."
            action={
              <Button
                appearance="brand"
                hierarchy="primary"
                size="md"
                label="Try again"
                onClick={() => {
                  void query.refetch();
                }}
              />
            }
          />
        </section>
      ) : null}
      {query.data?.empty ? (
        <section className={styles.empty}>
          <Text as="h2" variant="heading" size="sm" className={styles.panelTitle}>
            {query.data.has_history ? "Nothing in this range" : "No generations yet"}
          </Text>
          <Text as="p" variant="body" size="md">
            {query.data.has_history
              ? "No runs finished in this range. Choose a longer one to see earlier generations."
              : "Make a video from Home and this page will show what you’ve created, how long it took, and what to try next. Gen Space generations are not included."}
          </Text>
        </section>
      ) : null}
      {query.data && !query.data.empty ? (
        <DashboardPanels
          snapshot={query.data}
          width={width}
          layout={dashboardLayout(width)}
          selection={selection}
          onSelectionChange={setSelection}
        />
      ) : null}
    </div>
  );
}

function useDashboardWidth(ref: RefObject<HTMLDivElement | null>): number {
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const node = ref.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    setWidth(node.getBoundingClientRect().width);
    const observer = new ResizeObserver((entries) => {
      setWidth(entries[0]?.contentRect.width ?? 0);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [ref]);
  return width;
}
