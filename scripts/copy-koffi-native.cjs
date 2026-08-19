#!/usr/bin/env node
// electron-builder afterPack: copy Koffi's platform .node next to the app so
// packaged Windows (and other OS) builds can load it.
//
// koffi 3.x keeps the native addon in an optional sibling package
// (@koromix/koffi-${platform}-${arch}), not inside node_modules/koffi. Packing
// koffi/** therefore ships JS/headers only. At runtime Koffi searches
// process.resourcesPath/koffi/${platform}_${arch}/koffi.node — copy there.
// electron-builder also asarUnpacks node_modules/@koromix/** as a second path.
'use strict'

const fs = require('node:fs')
const path = require('node:path')
const { createRequire } = require('node:module')

const ARCH_NAMES = ['ia32', 'x64', 'armv7l', 'arm64', 'universal']

function archName(arch) {
  if (typeof arch === 'string') {
    return arch
  }
  return ARCH_NAMES[arch] ?? String(arch)
}

function koffiNativeDest(resourcesDir, platform, arch) {
  const triplet = `${platform}_${archName(arch)}`
  return path.join(resourcesDir, 'koffi', triplet, 'koffi.node')
}

function resourcesDir(context) {
  const packager = context.packager
  if (typeof packager.getResourcesDir === 'function') {
    return packager.getResourcesDir(context.appOutDir)
  }
  if (context.electronPlatformName === 'darwin') {
    const name = packager.appInfo.productFilename
    return path.join(context.appOutDir, `${name}.app`, 'Contents', 'Resources')
  }
  return path.join(context.appOutDir, 'resources')
}

function resolveKoffiNativeSrc(projectDir, platform, arch) {
  const pkgName = `@koromix/koffi-${platform}-${archName(arch)}`
  // koffi's exports map does not include ./package.json — resolve the entry.
  const fromProject = createRequire(path.join(projectDir, 'package.json'))
  const koffiEntry = fromProject.resolve('koffi')
  const fromKoffi = createRequire(koffiEntry)
  const nativeEntry = fromKoffi.resolve(pkgName)
  const src = path.join(path.dirname(nativeEntry), `${platform}_${archName(arch)}`, 'koffi.node')
  if (!fs.existsSync(src)) {
    throw new Error(`Koffi native binary missing at ${src} (package ${pkgName})`)
  }
  return src
}

async function copyKoffiNative(context) {
  const platform = context.electronPlatformName
  const arch = archName(context.arch)
  const projectDir = context.packager.projectDir
  const dest = koffiNativeDest(resourcesDir(context), platform, arch)
  const src = resolveKoffiNativeSrc(projectDir, platform, arch)

  fs.mkdirSync(path.dirname(dest), { recursive: true })
  fs.copyFileSync(src, dest)
  console.log(`[copy-koffi-native] ${src} → ${dest}`)
}

module.exports = {
  default: copyKoffiNative,
  archName,
  koffiNativeDest,
  resourcesDir,
  resolveKoffiNativeSrc,
  copyKoffiNative,
}
