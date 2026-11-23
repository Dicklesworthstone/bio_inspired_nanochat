#!/usr/bin/env bash
# Execute a checked-in recipe on the Linux DSR worker and retain its artifacts.
set -euo pipefail
case "${1:-}" in
  main)        event=push;     workflow=ci;                  port=34567 ;;
  validation)  event=schedule; workflow=nightly-validation;  port=34568 ;;
  uncertainty) event=schedule; workflow=nightly-uncertainty; port=34569 ;;
  *) echo "Expected main, validation or uncertainty" >&2; exit 2 ;;
esac

git rev-parse HEAD
# Match the Linux worker's writable tool cache. Override the image's XDG
# path into /home/runner, which belongs to another UID.
verifier=$(command -v minisign)
artifacts="$PWD/runs/dsr/${DSR_RUN_ID:?}/$1"
mkdir -p "$artifacts"
exec act "$event" --workflows ".github/workflows/$workflow.yml" \
  --platform ubuntu-latest=catthehacker/ubuntu:full-22.04 \
  --container-options "--user=1000:1000 --volume=$verifier:/usr/local/bin/minisign:ro" \
  --env XDG_CONFIG_HOME=/tmp/dsr-xdg --bind=false --concurrent-jobs 2 \
  --env-file /dev/null --secret-file /dev/null --var-file /dev/null \
  --artifact-server-path "$artifacts" \
  --artifact-server-port "$port" --pull=false
