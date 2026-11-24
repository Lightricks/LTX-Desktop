import { useEffect } from "react";
import { useNavigate } from "react-router";

import { startDeepLinkCourier } from "../lib/deep-link-courier";
import { pathForDeepLinkTarget } from "../lib/deep-link-path";
import { logger } from "../lib/logger";

export function useDeepLinkCourier(): void {
  const navigate = useNavigate();

  useEffect(() => {
    return startDeepLinkCourier({
      subscribe: (listener) =>
        window.electronAPI.onDeepLink((intent) => listener(intent)),
      take: () => window.electronAPI.takePendingDeepLink(),
      navigate,
      pathForTarget: pathForDeepLinkTarget,
      logError: (message) => logger.error(message),
    });
  }, [navigate]);
}
