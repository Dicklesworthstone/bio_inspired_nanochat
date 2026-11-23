#!/usr/bin/env bash
# Install one signed, version-matched UBS release into a private DSR job.
set -euo pipefail
release=https://github.com/Dicklesworthstone/ultimate_bug_scanner/releases/download/v5.5.26
key='RWS+jJ7psytzl3v4znpraY9VWBQrICXBFmT3VwvxpTzbuV2Q/CBTDmVJ'
staging=$(mktemp -d)
cd "$staging"
export CURL_HOME="$staging"
printf '%s\n' 'user-agent = "bio_inspired_nanochat-agent (+https://github.com/Dicklesworthstone/bio_inspired_nanochat)"' > .curlrc
for asset in SHA256SUMS SHA256SUMS.minisig install.sh; do
  curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location "$release/$asset" -o "$asset"
done
minisign -Vm SHA256SUMS -P "$key" -x SHA256SUMS.minisig
awk '$2 == "install.sh" {print; count++} END {if (count != 1) exit 1}' SHA256SUMS > installer.sha256
sha256sum --check installer.sha256
# A fresh job-private cache avoids sharing a partly installed release with any
# other job. The signed runner verifies its language modules against embedded hashes.
export XDG_CACHE_HOME="$staging/cache" UBS_ARTIFACT_BASE="$release" UBS_MINISIGN_PUBKEY="$key"
bash install.sh --non-interactive
printf 'XDG_CACHE_HOME=%s\n' "$XDG_CACHE_HOME" >> "${GITHUB_ENV:?}"
printf '%s\n' "$HOME/.local/bin" >> "${GITHUB_PATH:?}"
