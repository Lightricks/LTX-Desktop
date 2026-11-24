// Lightweight error logging for the design system: console only, no analytics or
// monitoring. Keeps `frontend/ds/` self-contained.

export type SeverityLevel = "fatal" | "error" | "warning" | "log" | "info" | "debug";

const isDevEnv = () => import.meta.env.MODE !== "production";

export function getErrorMessage(err: unknown): string {
  return err instanceof Error ? err.message : "Something went wrong";
}

export function handleErrors(errors: unknown[]): void {
  if (!isDevEnv()) {
    return;
  }
  for (const err of errors) {
    console.error(err);
  }
}

export function logMessage(message: string, level: SeverityLevel = "info"): void {
  if (!isDevEnv()) {
    return;
  }
  if (level === "error" || level === "fatal") {
    console.error(message);
  } else if (level === "warning") {
    console.warn(message);
  } else {
    console.trace(message);
  }
}
