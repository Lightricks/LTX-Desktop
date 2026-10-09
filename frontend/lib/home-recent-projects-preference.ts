/**
 * Browser-local list of recently opened Gen Space projects for the sidebar.
 *
 * Recorded when a project is opened. Most-recent-first, capped at
 * {@link MAX_SIDEBAR_RECENT_ITEMS}, and entries older than
 * {@link RECENT_SIDEBAR_PROJECT_TTL_MS} since last open are dropped.
 */

import {
  MAX_SIDEBAR_RECENT_ITEMS,
  RECENT_SIDEBAR_PROJECT_TTL_MS,
} from "./home-sidebar-recents.ts";
import { readProject } from "./project-storage.ts";

export const MAX_RECENT_PROJECTS = MAX_SIDEBAR_RECENT_ITEMS;
const STORAGE_KEY = "ltx-desktop.sidebarRecentProjects";

export const EMPTY_RECENT_PROJECT_IDS: readonly string[] = [];

export type RecentProjectSidebarItem = {
  readonly id: string;
  readonly name: string;
};

type RecentProjectRecord = {
  readonly id: string;
  readonly lastOpenedAt: number;
};

type Listener = () => void;

const EMPTY_RESOLVED: readonly RecentProjectSidebarItem[] = [];

const listeners = new Set<Listener>();

let cachedRecords: RecentProjectRecord[] = [];
let cachedIds: readonly string[] = EMPTY_RECENT_PROJECT_IDS;
let cachedResolved: readonly RecentProjectSidebarItem[] = EMPTY_RESOLVED;
let storageListenerAttached = false;

function notifyListeners(): void {
  for (const listener of listeners) {
    listener();
  }
}

function sameRecords(
  a: readonly RecentProjectRecord[],
  b: readonly RecentProjectRecord[],
): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  return a.every(
    (entry, index) =>
      entry.id === b[index]?.id &&
      entry.lastOpenedAt === b[index]?.lastOpenedAt,
  );
}

function sameIds(a: readonly string[], b: readonly string[]): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  return a.every((id, index) => id === b[index]);
}

function sameResolved(
  a: readonly RecentProjectSidebarItem[],
  b: readonly RecentProjectSidebarItem[],
): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  return a.every(
    (item, index) => item.id === b[index]?.id && item.name === b[index]?.name,
  );
}

function isExpired(record: RecentProjectRecord, now = Date.now()): boolean {
  return now - record.lastOpenedAt > RECENT_SIDEBAR_PROJECT_TTL_MS;
}

function parseStoredRecords(parsed: unknown): {
  records: RecentProjectRecord[];
  migrated: boolean;
} {
  if (!Array.isArray(parsed)) return { records: [], migrated: false };
  const now = Date.now();
  const records: RecentProjectRecord[] = [];
  let migrated = false;
  for (const entry of parsed) {
    if (typeof entry === "string" && entry.length > 0) {
      migrated = true;
      records.push({ id: entry, lastOpenedAt: now });
      continue;
    }
    if (
      entry &&
      typeof entry === "object" &&
      "id" in entry &&
      typeof entry.id === "string" &&
      entry.id.length > 0 &&
      "lastOpenedAt" in entry &&
      typeof entry.lastOpenedAt === "number"
    ) {
      records.push({ id: entry.id, lastOpenedAt: entry.lastOpenedAt });
    }
  }
  return { records, migrated };
}

function readStoredRecords(): {
  records: RecentProjectRecord[];
  migrated: boolean;
} {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { records: [], migrated: false };
    return parseStoredRecords(JSON.parse(raw));
  } catch {
    return { records: [], migrated: false };
  }
}

function normalizeRecords(
  records: readonly RecentProjectRecord[],
  now = Date.now(),
): RecentProjectRecord[] {
  const seen = new Set<string>();
  const next: RecentProjectRecord[] = [];
  for (const record of records) {
    if (seen.has(record.id)) continue;
    if (isExpired(record, now)) continue;
    if (readProject(record.id) == null) continue;
    seen.add(record.id);
    next.push(record);
    if (next.length >= MAX_RECENT_PROJECTS) break;
  }
  return next;
}

function idsOf(records: readonly RecentProjectRecord[]): readonly string[] {
  if (records.length === 0) return EMPTY_RECENT_PROJECT_IDS;
  return records.map((record) => record.id);
}

function resolveRecords(
  records: readonly RecentProjectRecord[],
): readonly RecentProjectSidebarItem[] {
  if (records.length === 0) return EMPTY_RESOLVED;
  return records.flatMap((record) => {
    const project = readProject(record.id);
    if (!project) return [];
    return [{ id: record.id, name: project.name.trim() || "Untitled project" }];
  });
}

function commit(records: RecentProjectRecord[], notify: boolean): void {
  const recordsChanged = !sameRecords(cachedRecords, records);
  const nextIds = idsOf(records);
  const idsChanged = !sameIds(cachedIds, nextIds);
  const nextResolved = resolveRecords(records);
  const resolvedChanged = !sameResolved(cachedResolved, nextResolved);
  if (!recordsChanged && !idsChanged && !resolvedChanged) return;
  if (recordsChanged) cachedRecords = records;
  if (idsChanged) cachedIds = nextIds;
  if (resolvedChanged) cachedResolved = nextResolved;
  if (notify && (idsChanged || resolvedChanged)) notifyListeners();
}

function persistRecords(records: readonly RecentProjectRecord[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  } catch {
    // Storage can be unavailable in restricted contexts.
  }
}

/**
 * Load storage into memory. Id-only rows are rewritten once as
 * `{ id, lastOpenedAt }` so the TTL clock starts instead of resetting on
 * every read.
 */
function loadFromStorage(): RecentProjectRecord[] {
  const stored = readStoredRecords();
  const normalized = normalizeRecords(stored.records);
  if (stored.migrated || !sameRecords(stored.records, normalized)) {
    persistRecords(normalized);
  }
  return normalized;
}

/** With no subscriber, storage is the source of truth (cold load, tests). */
function syncFromStorageIfUnsubscribed(): void {
  if (listeners.size > 0) return;
  commit(loadFromStorage(), false);
}

function writeRecords(records: RecentProjectRecord[]): void {
  persistRecords(records);
  commit(records, true);
}

function ensureStorageListener(): void {
  if (storageListenerAttached || typeof window === "undefined") return;
  storageListenerAttached = true;
  window.addEventListener("storage", (event) => {
    if (event.key !== null && event.key !== STORAGE_KEY) return;
    commit(loadFromStorage(), true);
  });
}

function refreshResolvedNames(): void {
  const next = resolveRecords(cachedRecords);
  if (!sameResolved(cachedResolved, next)) cachedResolved = next;
}

/**
 * Most-recent-first project ids (max {@link MAX_RECENT_PROJECTS}).
 * Pure snapshot: never reads or writes storage. Hydration happens in
 * {@link subscribeRecentProjectIds} and {@link recordRecentProjectId}.
 */
export function getRecentProjectIds(): readonly string[] {
  return cachedIds;
}

/**
 * Sidebar rows with live project titles; omits deleted and expired projects.
 * May refresh names from project storage, and must not persist.
 */
export function getResolvedRecentProjects(): readonly RecentProjectSidebarItem[] {
  refreshResolvedNames();
  return cachedResolved;
}

export function recordRecentProjectId(projectId: string): void {
  if (!projectId) return;
  if (!readProject(projectId)) return;
  syncFromStorageIfUnsubscribed();
  const now = Date.now();
  const next = normalizeRecords(
    [
      { id: projectId, lastOpenedAt: now },
      ...cachedRecords.filter((record) => record.id !== projectId),
    ],
    now,
  );
  if (sameRecords(cachedRecords, next)) return;
  writeRecords(next);
}

export function removeRecentProjectId(projectId: string): void {
  syncFromStorageIfUnsubscribed();
  const next = cachedRecords.filter((record) => record.id !== projectId);
  if (sameRecords(cachedRecords, next)) return;
  writeRecords(next);
}

export function subscribeRecentProjectIds(listener: Listener): () => void {
  ensureStorageListener();
  if (listeners.size === 0) {
    commit(loadFromStorage(), false);
  }
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Test helper — wipe the local list. */
export function clearRecentProjectIds(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
  commit([], true);
}
