import { create } from "zustand";

type FeatureFormMuteStore = {
  isMuted: boolean;
  setMuted: (isMuted: boolean) => void;
};

/**
 * Session-wide playback mute. Same as ltx.io: start muted, keep the last
 * toggle across results, tiles, and lightbox items.
 */
export const useFeatureFormMuteStore = create<FeatureFormMuteStore>((set) => ({
  isMuted: true,
  setMuted: (isMuted) => set({ isMuted }),
}));
