import eslintConfigPrettier from "eslint-config-prettier/flat";
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

// Minimal, boundary-focused ESLint config. It intentionally enables no broad rule
// sets — it enforces design-system import boundaries and the Electron host split.
export default [
  {
    ignores: ["frontend/generated/**"],
  },
  {
    // Don't fail on the codebase's existing `eslint-disable react-hooks/*` directives
    // (which are inert here — the rule is registered below but left off).
    linterOptions: {
      reportUnusedDisableDirectives: "off",
    },
  },
  {
    files: ["frontend/**/*.{ts,tsx}"],
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
    },
    // Registered so existing `react-hooks/*` disable comments reference a known rule.
    // The rules themselves stay off — this config only enforces import boundaries.
    plugins: {
      "react-hooks": reactHooks,
    },
  },
  {
    // The DS itself, ltx-io, Home, and the LAN remote SPA are approved
    // consumers of the vendored design-system primitives.
    files: ["frontend/**/*.{ts,tsx}"],
    ignores: [
      "frontend/ds/**",
      "frontend/ltx-io/**",
      "frontend/components/home/**",
      "frontend/views/home/**",
      "frontend/remote/**",
    ],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              // Both aliases resolve to frontend/ds, so block both spellings.
              group: ["@ds", "@ds/**", "@/ds", "@/ds/**"],
              message:
                "Desktop app code must not depend on the vendored ltx.io design system (@ds, @/ds).",
            },
          ],
        },
      ],
    },
  },
  {
    // Product-coupled ltx-io surfaces remain unavailable to most Desktop code.
    // ltx-io itself, the LAN remote SPA, and the Home feature registry (the
    // single Desktop/Remote screen map) may import product-coupled surfaces.
    files: ["frontend/**/*.{ts,tsx}"],
    ignores: [
      "frontend/ds/**",
      "frontend/ltx-io/**",
      "frontend/remote/**",
      "frontend/lib/home-feature-registry.ts",
    ],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["@/ltx-io", "@/ltx-io/**"],
              message:
                "Desktop app code must not depend on product-coupled ltx-io surfaces.",
            },
          ],
        },
      ],
    },
  },
  {
    // The design system must stay self-contained so it remains extractable:
    // only the `@ds` alias, relative paths, and npm packages are allowed — never
    // reach up into the rest of the app via `@/…`.
    files: ["frontend/ds/**/*.{ts,tsx}"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["@/**"],
              message:
                "ds/ must stay self-contained: import only from @ds, relative paths, or npm packages.",
            },
          ],
        },
      ],
    },
  },
  {
    // Dual-host Home/Explore/Remote surfaces must stay Electron-free. Desktop
    // adapters in those trees are ignored so editor/settings/setup (outside
    // these globs) and host-specific modules can still use window.electronAPI.
    files: [
      "frontend/ltx-io/**/*.{ts,tsx}",
      "frontend/components/home/**/*.{ts,tsx}",
      "frontend/remote/**/*.{ts,tsx}",
    ],
    ignores: [
      "frontend/ltx-io/runtime/createDesktopExploreRuntime.ts",
      "frontend/components/home/home-external-links.ts",
      "frontend/components/home/home-external-links.test.ts",
    ],
    rules: {
      "no-restricted-syntax": [
        "error",
        {
          selector:
            "MemberExpression[property.name='electronAPI'], MemberExpression[property.value='electronAPI']",
          message:
            "Dual-host Home/Explore/Remote code must not access window.electronAPI. Put Electron calls in a Desktop adapter.",
        },
        {
          selector: "ImportDeclaration[source.value='electron']",
          message:
            "Dual-host Home/Explore/Remote code must not import Electron. Put Electron calls in a Desktop adapter.",
        },
      ],
    },
  },
  eslintConfigPrettier,
];
