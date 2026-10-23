#!/usr/bin/env bash
# Keep CI independent of GitHub's release-list API, which can return an empty page.
set -euo pipefail

: "${RUNNER_TEMP:?GitHub Actions temporary directory is required}"
: "${GITHUB_PATH:?GitHub Actions path file is required}"
if [[ "$(uname -sm)" != "Linux x86_64" ]]; then
  echo "The pinned CI protoc archive requires Linux x86_64" >&2
  exit 1
fi

protoc_dir=$(mktemp -d "$RUNNER_TEMP/hades-protoc.XXXXXX")
curl --fail --location --retry 3 --silent --show-error \
  --output "$protoc_dir/protoc.zip" \
  https://github.com/protocolbuffers/protobuf/releases/download/v23.4/protoc-23.4-linux-x86_64.zip
printf '%s  %s\n' \
  0502f286ac9ed860b629a7965a14527b1f2dd131e4283fa23c2d7f184672aa9a \
  "$protoc_dir/protoc.zip" | sha256sum --check --strict
unzip -q "$protoc_dir/protoc.zip" -d "$protoc_dir"
rm "$protoc_dir/protoc.zip"
test "$("$protoc_dir/bin/protoc" --version)" = "libprotoc 23.4"
printf '%s\n' "$protoc_dir/bin" >> "$GITHUB_PATH"
