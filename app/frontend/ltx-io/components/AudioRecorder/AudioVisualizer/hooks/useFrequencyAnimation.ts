import { useCallback, useEffect, useRef } from "react";

export interface UseFrequencyAnimationOptions {
  /** Callback to process frequency data on each animation frame */
  onFrame: (frequencyData: Uint8Array) => void;
  /** Callback when animation is stopped (for drawing idle state) */
  onIdle?: () => void;
}

/**
 * Hook to run an animation loop that captures frequency data from an AnalyserNode.
 * Automatically starts/stops based on isActive state.
 */
export function useFrequencyAnimation(
  analyser: AnalyserNode | null,
  isActive: boolean,
  options: UseFrequencyAnimationOptions,
): void {
  const { onFrame, onIdle } = options;
  const animationFrameRef = useRef<number | null>(null);

  const animate = useCallback(() => {
    if (!analyser) return;

    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    onFrame(data);

    animationFrameRef.current = requestAnimationFrame(animate);
  }, [analyser, onFrame]);

  useEffect(() => {
    if (isActive && analyser) {
      animate();
    } else {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      onIdle?.();
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [analyser, isActive, animate, onIdle]);
}
