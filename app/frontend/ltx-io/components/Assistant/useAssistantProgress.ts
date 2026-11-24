import { createParser } from "eventsource-parser";
import { useEffect, useRef, useState } from "react";
import { z } from "zod";

import type { ExploreApi } from "@/ltx-io/runtime/ExploreRuntime";

const progressSchema = z.object({
  active: z.boolean(),
  activity: z.string(),
  text: z.string(),
  done: z.boolean(),
});
const IDLE = { active: false, activity: "", text: "" };

/**
 * `resumedAt` reopens the stream after it ended, when the same run is continued.
 * When a step ends its text is kept, and `onStepEnd` fetches the reply that replaces it.
 */
export function useAssistantProgress(
  stream: ExploreApi["streamAssistantEvents"],
  runId: string | null,
  resumedAt = 0,
  onStepEnd?: () => void,
) {
  const [progress, setProgress] = useState(IDLE);
  const streamRef = useRef(stream);
  streamRef.current = stream;
  const onStepEndRef = useRef(onStepEnd);
  onStepEndRef.current = onStepEnd;
  useEffect(() => {
    setProgress(IDLE);
    if (!runId) return;
    const abort = new AbortController();
    let retry: ReturnType<typeof setTimeout>;
    let done = false;
    let wasActive = false;
    let delay = 1500;
    const connect = async () => {
      try {
        const response = await streamRef.current(runId, { signal: abort.signal });
        // A client error will not fix itself; polling still settles the run.
        if (response.status >= 400 && response.status < 500) done = true;
        if (!response.ok || !response.body) throw new Error("Progress unavailable");
        delay = 1500;
        const parser = createParser({
          onEvent: ({ data }) => {
            const parsed = progressSchema.safeParse(JSON.parse(data));
            if (!parsed.success || abort.signal.aborted) return;
            done = parsed.data.done;
            const { active, activity, text } = parsed.data;
            if (active) setProgress({ active, activity, text });
            else setProgress((current) => ({ active, activity, text: current.text }));
            if (wasActive && !active) onStepEndRef.current?.();
            wasActive = active;
          },
        });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        try {
          while (!abort.signal.aborted && !done) {
            const { value, done: ended } = await reader.read();
            if (ended) break;
            parser.feed(decoder.decode(value, { stream: true }));
          }
        } finally {
          reader.releaseLock();
        }
      } catch {
        /* The persisted run remains recoverable when its progress stream disconnects. */
      }
      if (abort.signal.aborted || done) return;
      retry = setTimeout(() => void connect(), delay);
      delay = Math.min(delay * 2, 30_000);
    };
    void connect();
    return () => {
      abort.abort();
      clearTimeout(retry);
    };
  }, [runId, resumedAt]);
  return progress;
}
