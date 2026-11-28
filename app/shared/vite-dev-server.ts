const DEFAULT_VITE_DEV_SERVER_PORT = 5173
const MAX_TCP_PORT = 65535

export type EnvLike = {
  readonly [key: string]: string | undefined
}

export function resolveViteDevServerPort(env: EnvLike = process.env): number {
  const raw = env.VITE_DEV_SERVER_PORT
  if (typeof raw !== 'string' || !/^[0-9]+$/.test(raw)) {
    return DEFAULT_VITE_DEV_SERVER_PORT
  }

  const port = Number(raw)
  if (!Number.isInteger(port) || port < 1 || port > MAX_TCP_PORT) {
    return DEFAULT_VITE_DEV_SERVER_PORT
  }

  return port
}

export function getViteDevServerUrl(env: EnvLike = process.env): string {
  return `http://localhost:${resolveViteDevServerPort(env)}`
}
