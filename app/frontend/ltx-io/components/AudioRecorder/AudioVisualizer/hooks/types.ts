export interface AnalyserConfig {
  fftSize?: number;
  minDecibels?: number;
  maxDecibels?: number;
  smoothingTimeConstant?: number;
}

export interface AnalyserState {
  analyser: AnalyserNode | null;
  isActive: boolean;
  resume: () => Promise<void>;
}
