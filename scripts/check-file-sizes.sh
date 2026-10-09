#!/usr/bin/env bash

# File size checker for added, modified, and renamed media files.
# Usage: bash scripts/check-file-sizes.sh [base-branch]
# Default base branch: origin/main

set -euo pipefail

readonly MAX_VIDEO_SIZE=$((5000 * 1024))
readonly MAX_IMAGE_SIZE=$((500 * 1024))
readonly MAX_AUDIO_SIZE=$((500 * 1024))
readonly DEFAULT_BASE_BRANCH="origin/main"

human_readable_size() {
  local size="$1"
  if ((size >= 1024 * 1024)); then
    echo "$((size / 1024 / 1024)) MB"
  elif ((size >= 1024)); then
    echo "$((size / 1024)) KB"
  else
    echo "$size bytes"
  fi
}

get_extension() {
  echo "${1##*.}" | tr '[:upper:]' '[:lower:]'
}

get_file_limit() {
  local extension
  extension=$(get_extension "$1")

  case "$extension" in
    mp4|webm|avi|mov|mkv|flv|wmv|m4v|3gp|ogv)
      echo "$MAX_VIDEO_SIZE"
      ;;
    jpg|jpeg|png|gif|bmp|webp|svg|tiff|tif|ico|avif)
      echo "$MAX_IMAGE_SIZE"
      ;;
    mp3|wav|flac|aac|ogg|wma|m4a|opus)
      echo "$MAX_AUDIO_SIZE"
      ;;
    *)
      echo "0"
      ;;
  esac
}

get_file_type() {
  local extension
  extension=$(get_extension "$1")

  case "$extension" in
    mp4|webm|avi|mov|mkv|flv|wmv|m4v|3gp|ogv)
      echo "video"
      ;;
    jpg|jpeg|png|gif|bmp|webp|svg|tiff|tif|ico|avif)
      echo "image"
      ;;
    mp3|wav|flac|aac|ogg|wma|m4a|opus)
      echo "audio"
      ;;
    *)
      echo "unknown"
      ;;
  esac
}

main() {
  local base_branch="${1:-$DEFAULT_BASE_BRANCH}"
  echo "Checking file sizes against limits..."
  echo "Base branch: $base_branch"
  echo "Limits: videos $(human_readable_size "$MAX_VIDEO_SIZE"), images/audio $(human_readable_size "$MAX_IMAGE_SIZE")"

  if ! git rev-parse --verify "$base_branch" >/dev/null 2>&1; then
    if [[ "$base_branch" == origin/* ]]; then
      git fetch origin "${base_branch#origin/}"
    else
      echo "Base branch '$base_branch' was not found."
      exit 1
    fi
  fi

  local changed_files
  changed_files=$(git diff --name-only --diff-filter=AMR "$base_branch"...HEAD 2>/dev/null || true)

  if [[ -z "$changed_files" ]]; then
    echo "No changed files found."
    exit 0
  fi

  local violations=()
  local total_files=0
  local checked_files=0

  while IFS= read -r file; do
    total_files=$((total_files + 1))
    [[ -f "$file" ]] || continue

    local size_limit
    size_limit=$(get_file_limit "$file")
    ((size_limit > 0)) || continue

    checked_files=$((checked_files + 1))
    local file_size
    file_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file")
    if ((file_size > size_limit)); then
      violations+=(
        "$file: $(human_readable_size "$file_size") $(get_file_type "$file"), limit $(human_readable_size "$size_limit")"
      )
    fi
  done <<< "$changed_files"

  echo "Checked $checked_files media files across $total_files changed files."

  if ((${#violations[@]} > 0)); then
    echo "Files exceeding size limits:"
    printf '  %s\n' "${violations[@]}"
    exit 1
  fi

  echo "All changed media files pass size limits."
}

main "$@"
