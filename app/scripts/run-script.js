#!/usr/bin/env node
// Runs a bash script on macOS/Linux or a PowerShell script on Windows.
// Usage: node scripts/run-script.js <base-name> [args...]
//
// Example: node scripts/run-script.js scripts/local-build --skip-python
//   macOS/Linux → bash scripts/local-build.sh --skip-python
//   Windows     → powershell -ExecutionPolicy Bypass -File scripts/local-build.ps1 -SkipPython
//
// Arg conversion for PowerShell: --foo-bar → -FooBar, --foo → -Foo

import { execSync } from 'child_process'

let [,, baseName, ...args] = process.argv

if (!baseName) {
  console.error('Usage: node scripts/run-script.js <script-base> [args...]')
  process.exit(1)
}

// Strip platform-specific extensions so tab-completed paths work
baseName = baseName.replace(/\.(sh|ps1)$/, '')

if (process.platform === 'win32') {
  // Convert --kebab-case args to -PascalCase for PowerShell
  const psArgs = args.map(arg => {
    if (!arg.startsWith('--')) return arg
    return '-' + arg.slice(2).split('-').map(w => w[0].toUpperCase() + w.slice(1)).join('')
  })
  const cmd = `powershell -ExecutionPolicy Bypass -File ${baseName}.ps1 ${psArgs.join(' ')}`
  execSync(cmd, { stdio: 'inherit' })
} else {
  const cmd = `bash ${baseName}.sh ${args.join(' ')}`
  execSync(cmd, { stdio: 'inherit' })
}
