import type { DeepLinkTarget } from "../../shared/deep-link.ts";

export type DeepLinkCourierIntent = {
  id: string;
  target: DeepLinkTarget;
};

export type DeepLinkCourierDeps = {
  subscribe: (listener: (intent: DeepLinkCourierIntent) => void) => () => void;
  take: () => Promise<DeepLinkCourierIntent | null>;
  navigate: (path: string) => void;
  pathForTarget: (target: DeepLinkTarget) => string;
  logError?: (message: string) => void;
};

export function startDeepLinkCourier(deps: DeepLinkCourierDeps): () => void {
  let cancelled = false;
  const seen = new Set<string>();

  const apply = (intent: DeepLinkCourierIntent | null) => {
    if (!intent || cancelled || seen.has(intent.id)) return;
    seen.add(intent.id);
    deps.navigate(deps.pathForTarget(intent.target));
  };

  const takeAndApply = () => {
    void deps
      .take()
      .then(apply)
      .catch((err: unknown) => {
        deps.logError?.(`Failed to take pending deep link: ${err}`);
      });
  };

  const unsubscribe = deps.subscribe(() => {
    takeAndApply();
  });
  takeAndApply();

  return () => {
    cancelled = true;
    unsubscribe();
  };
}
