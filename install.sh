#!/usr/bin/env bash
# Ouroboros — install all seven plugins into the active Hermes profile.
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/plugins"

mkdir -p "$HERMES_HOME/plugins"
for plugin in seatbelt echo archive council autopilot forge ouroboros; do
  rm -rf "${HERMES_HOME:?}/plugins/$plugin"
  cp -r "$SRC/$plugin" "$HERMES_HOME/plugins/$plugin"
  echo "  🐍 $plugin"
done

cat <<'EOF'

Ouroboros installed. Enable the pack:

  hermes plugins enable seatbelt echo council autopilot forge ouroboros
  hermes plugins enable archive

Archive is a context engine — also add to ~/.hermes/config.yaml:

  context:
    engine: archive

Then spin the flywheel:

  /ouroboros
EOF
