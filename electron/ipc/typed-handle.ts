import { ipcMain } from 'electron'
import { z } from 'zod'
import { electronAPISchemas } from '../../shared/electron-api-schema'

type Schemas = typeof electronAPISchemas

export function handle<K extends keyof Schemas>(
  key: K & string,
  handler: (
    input: z.infer<Schemas[K]['input']>,
  ) => Promise<z.infer<Schemas[K]['output']>> | z.infer<Schemas[K]['output']>,
): void {
  ipcMain.handle(key, (_event, input) => handler(input ?? {}))
}
