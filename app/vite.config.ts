import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import svgr from "vite-plugin-svgr";
import electron from "vite-plugin-electron";
import renderer from "vite-plugin-electron-renderer";
import path from "path";
import { optionalOverlayAliases } from "./shared/optional-overlay";
import { resolveViteDevServerPort } from "./shared/vite-dev-server";

function electronSpawnEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  // Cursor (and other Electron hosts) set this; if inherited, LTX Desktop
  // starts as Node and `require('electron')` is the npm path, not the APIs.
  delete env.ELECTRON_RUN_AS_NODE;
  return env;
}

export default defineConfig(({ command }) => {
  // vite-plugin-electron *builds* main/preload even during `vite` (serve).
  // Bind from this parent command so `pnpm dev` still overlays those entries;
  // `vite build` always keeps the public stubs.
  const overlayAliases = optionalOverlayAliases({ root: __dirname, command });

  return {
    server: {
      port: resolveViteDevServerPort(),
      strictPort: true,
    },
    css: {
      preprocessorOptions: {
        scss: {
          api: "modern-compiler",
          silenceDeprecations: ["import"],
        },
      },
    },
    plugins: [
      react(),
      svgr({ include: "**/*.svg?react" }),
      electron([
        {
          entry: "electron/main.ts",
          onstart(options) {
            const env = electronSpawnEnv();
            if (process.env.ELECTRON_DEBUG) {
              // --inspect and --remote-debugging-port must come before '.' (the app path)
              options.startup(
                [
                  "--inspect=9229",
                  "--remote-debugging-port=9222",
                  ".",
                  "--no-sandbox",
                ],
                { env },
              );
            } else {
              options.startup([".", "--no-sandbox"], { env });
            }
          },
          vite: {
            resolve: { alias: overlayAliases },
            build: {
              outDir: "dist-electron",
              sourcemap: true,
              rollupOptions: {
                external: ["electron", "koffi"],
              },
            },
          },
        },
        {
          entry: "electron/preload.ts",
          onstart(options) {
            options.reload();
          },
          vite: {
            resolve: { alias: overlayAliases },
            build: {
              outDir: "dist-electron",
              sourcemap: true,
              rollupOptions: {
                output: {
                  format: "cjs", // Preload must be CommonJS
                },
              },
            },
          },
        },
      ]),
      renderer(),
    ],
    resolve: {
      alias: [
        ...overlayAliases,
        // Dedicated alias for the vendored ltx.io design system so it can be
        // extracted later without a repo-wide import rewrite. Declared before '@'
        // (though '@' only matches '@/…', not '@ds/…').
        { find: "@ds", replacement: path.resolve(__dirname, "./frontend/ds") },
        { find: "@", replacement: path.resolve(__dirname, "./frontend") },
      ],
    },
    base: "./", // Use relative paths for Electron file:// protocol
    build: {
      outDir: "dist",
    },
  };
});
