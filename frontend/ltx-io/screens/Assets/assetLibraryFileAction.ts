import type { ExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";

import { folderRevealPath } from "../../runtime/folderRevealPath.ts";

export type AssetLibraryFileAction =
  | { kind: "reveal"; path: string }
  | { kind: "download"; url: string }
  | { kind: "none" };

export function resolveAssetLibraryFileAction({
  revealInFolder,
  path,
  mediaUrl,
}: {
  revealInFolder: ExploreRuntime["revealInFolder"];
  path: string | null | undefined;
  mediaUrl: string | null;
}): AssetLibraryFileAction {
  const revealPath = folderRevealPath(revealInFolder, path);
  if (revealPath !== null) {
    return { kind: "reveal", path: revealPath };
  }
  if (mediaUrl !== null) {
    return { kind: "download", url: mediaUrl };
  }
  return { kind: "none" };
}
