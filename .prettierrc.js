/**
 * Copied from LTX Studio `infinity-formatter` (vl-web-client).
 *
 * @type {import("prettier").Config}
 */
export default {
  importOrder: [
    // 1. All global packages from node_modules (react, lodash, etc.)
    "^(?!@/|\\.).*$",
    // 2. All global imports starting with @/
    "^@/",
    // 3. Relative imports from parent directories
    "^../",
    // 4. Relative imports from current directory
    "./",
  ],
  importOrderParserPlugins: ["typescript", "jsx", "decorators-legacy"],
  importOrderSeparation: true,
  importOrderSortSpecifiers: true,
  printWidth: 90,
  arrowParens: "always",
  objectWrap: "preserve",
  plugins: ["@trivago/prettier-plugin-sort-imports"],
};
