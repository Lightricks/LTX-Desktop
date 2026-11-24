type ActiveHoverVideo = {
  assetId: string;
  stop: () => void;
};

let activeHoverVideo: ActiveHoverVideo | null = null;

export function activateHoverVideo(assetId: string, stop: () => void): void {
  if (activeHoverVideo?.assetId !== assetId) {
    activeHoverVideo?.stop();
  }
  activeHoverVideo = { assetId, stop };
}

export function deactivateHoverVideo(assetId: string): void {
  if (activeHoverVideo?.assetId === assetId) {
    activeHoverVideo = null;
  }
}

export function resetHoverVideoCoordinator(): void {
  activeHoverVideo = null;
}
