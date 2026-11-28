import { useEffect, useRef } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router";
import { useAppChromeControls } from "../components/AppShell";
import {
  parseSettingsInitialReason,
  parseSettingsScrollAnchor,
  parseSettingsTabId,
} from "../lib/settings-navigation";
import { paths } from "../paths";

function stripHash(hash: string): string {
  return hash.startsWith("#") ? hash.slice(1) : hash;
}

function sectionFromLocation(hash: string, tabQuery: string | null) {
  const fromHash = parseSettingsTabId(stripHash(hash));
  if (fromHash) return fromHash;
  return parseSettingsTabId(tabQuery);
}

/**
 * Legacy hash links name a row, not a tab (#promptEnhancer was its own tab before the
 * split). parseSettingsTabId maps it to the tab that now owns the row; this keeps the row
 * itself so the modal can scroll to it instead of just landing on General.
 */
function scrollAnchorFromLocation(hash: string, tabQuery: string | null) {
  return (
    parseSettingsScrollAnchor(stripHash(hash)) ?? parseSettingsScrollAnchor(tabQuery)
  );
}

/** Deep links to /settings open the global modal and return to Home. */
export function SettingsPage() {
  const { hash } = useLocation();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { openSettings } = useAppChromeControls();
  const openedRef = useRef(false);

  useEffect(() => {
    if (openedRef.current) return;
    openedRef.current = true;
    openSettings({
      tab: sectionFromLocation(hash, searchParams.get("tab")),
      scrollAnchor: scrollAnchorFromLocation(hash, searchParams.get("tab")),
      reason: parseSettingsInitialReason(searchParams.get("reason")),
    });
    navigate(paths.home, { replace: true });
  }, [hash, navigate, openSettings, searchParams]);

  return null;
}
