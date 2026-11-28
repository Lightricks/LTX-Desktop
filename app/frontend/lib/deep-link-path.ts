import { generatePath } from "react-router";

import type { DeepLinkTarget } from "../../shared/deep-link.ts";
import { paths } from "../paths.ts";

import { HOME_SHELL_ENTRIES } from "./home-shell.ts";

export function pathForDeepLinkTarget(target: DeepLinkTarget): string {
  switch (target.kind) {
    case "fetcher-tool":
      return generatePath(paths.fetcherTool, { tool: target.slug });
    case "home-shell": {
      const entry = HOME_SHELL_ENTRIES.find((candidate) => candidate.id === target.slug);
      if (entry === undefined || entry.path === null) {
        throw new Error(`Unknown home-shell deep link: ${target.slug}`);
      }
      return entry.path;
    }
    default: {
      const _exhaustive: never = target;
      return _exhaustive;
    }
  }
}
