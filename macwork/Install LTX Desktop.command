#!/bin/bash
# Per-user Apple silicon installer. The native Mac download is pinned to its release digest.
set -euo pipefail
umask 077
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
here="$(cd "$(dirname "$0")" && pwd -P)"
if [[ "$(uname -s)" != Darwin ]]; then
  echo "Run this installer in macOS on an Apple Silicon Mac." >&2
  exit 1
fi
# Finder or Terminal may itself be running under Rosetta.
if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || true)" != 1 ]]; then
  echo "This app requires an Apple Silicon Mac." >&2
  exit 1
fi
if [[ "$(uname -m)" != arm64 ]]; then
  exec /usr/bin/arch -arm64 /bin/bash "$0" "$@"
fi
os_version="$(sw_vers -productVersion)"
if (( ${os_version%%.*} < 14 )); then
  echo "The community app requires macOS 14 or newer. This Mac runs $os_version." >&2
  exit 1
fi

apps="$HOME/Applications"
cache="$HOME/Library/Caches/com.spacespeak.ltx-desktop-community/setup/1.3.3"
launch=1
while (( $# )); do
  case "$1" in
    --cache) cache="$2"; shift 2 ;;
    --applications-dir) apps="$2"; shift 2 ;;
    --no-launch) launch=0; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
product="LTX Desktop - SpaceSpeak"
running="$(osascript -l JavaScript -e 'ObjC.import("AppKit"); $.NSRunningApplication.runningApplicationsWithBundleIdentifier("com.spacespeak.ltx-desktop-community").count')"
if [[ "$running" != 0 ]]; then
  echo "Quit the running $product app, then run this installer again." >&2
  exit 1
fi
mkdir -p "$apps" "$cache"
apps="$(cd "$apps" && pwd -P)"
cache="$(cd "$cache" && pwd -P)"
stage="$(mktemp -d "$apps/.ltx-spacespeak.XXXXXXXX")"
partial=""
cleanup() {
  [[ -z "$partial" ]] || rm -f -- "$partial"
  rm -rf -- "$stage"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fetch() {
  local name="$1" url="$2" expected="$3" actual
  if [[ -f "$cache/$name" ]]; then
    actual="$(shasum -a 256 "$cache/$name")"
    if [[ "${actual%% *}" = "$expected" ]]; then
      echo "Verified cached $name"
      return
    fi
  fi
  partial="$(mktemp "$cache/.download.XXXXXXXX")"
  echo "Downloading $name"
  curl --disable --proto '=https' --proto-redir '=https' --fail --location --show-error --retry 3 --output "$partial" "$url"
  actual="$(shasum -a 256 "$partial")"
  if [[ "${actual%% *}" != "$expected" ]]; then
    echo "$name failed SHA-256 verification." >&2
    exit 1
  fi
  mv -f "$partial" "$cache/$name"
  partial=""
}

fetch upstream-mac.zip 'https://github.com/Lightricks/LTX-Desktop/releases/download/v1.3.3/LTX-Desktop-arm64.zip' \
  90bad9cc1662e52f3d291ca2e44e2bee5651c5497e1720477317a6c616db7f51
echo "Preparing the native Mac runtime and desktop"
ditto -x -k "$cache/upstream-mac.zip" "$stage/upstream"
donor="$stage/upstream/LTX Desktop.app/Contents/Resources"
bundle="$stage/$product.app"
resources="$bundle/Contents/Resources"
mkdir -p "$resources"
"$donor/python/bin/python3" -I -B "$here/materialize.py" assemble "$here/community.zip" "$donor" "$bundle"
mv "$donor/python" "$resources/python"
echo "Verifying the signed Electron frameworks and helper apps"
for nested in "$bundle/Contents/Frameworks/"*.framework "$bundle/Contents/Frameworks/"*.app; do
  if ! codesign --verify --deep --strict --verbose=1 "$nested"; then
    echo "The signature of ${nested##*/} could not be verified in the staged Mac application." >&2
    exit 1
  fi
done

echo "Signing the community bundle locally and checking the installation"
codesign --force --sign - --identifier com.spacespeak.ltx-desktop-community --timestamp=none --entitlements "$here/entitlements.mac.plist" "$bundle"
codesign --verify --deep --strict "$bundle"
"$resources/python/bin/python3" -I -B "$here/check_native_mps.py" --backend "$resources/backend"
"$resources/python/bin/python3" -I -B "$here/materialize.py" activate "$bundle" "$apps"
echo "Installed $apps/$product.app"
if [[ "$launch" = 1 ]]; then open "$apps/$product.app"; fi
