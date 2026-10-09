/** PCM 16-bit WAV encoder. Used to turn a mic recording into an ingestible file. */
export function encodePcmWav(
  channelData: readonly Float32Array[],
  sampleRate: number,
): ArrayBuffer {
  if (channelData.length === 0) {
    throw new Error("Audio recording was empty.");
  }
  const numChannels = channelData.length;
  const samples = channelData[0]?.length ?? 0;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = samples * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples; i += 1) {
    for (let channel = 0; channel < numChannels; channel += 1) {
      const sample = channelData[channel]?.[i] ?? 0;
      const clipped = Math.min(1, Math.max(-1, sample));
      view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
      offset += 2;
    }
  }

  return buffer;
}

function writeAscii(view: DataView, offset: number, value: string): void {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}

export type RecordedWav = {
  wav: File;
  buffer: AudioBuffer;
};

export async function recordedAudioToWav(file: File): Promise<RecordedWav> {
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    const channels = Array.from({ length: decoded.numberOfChannels }, (_, index) =>
      decoded.getChannelData(index),
    );
    const wav = encodePcmWav(channels, decoded.sampleRate);
    return {
      wav: new File([wav], `recording-${Date.now()}.wav`, { type: "audio/wav" }),
      buffer: decoded,
    };
  } finally {
    await context.close();
  }
}

export async function recordedAudioToWavFile(file: File): Promise<File> {
  const { wav } = await recordedAudioToWav(file);
  return wav;
}
