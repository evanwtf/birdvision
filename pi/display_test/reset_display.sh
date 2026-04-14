#!/usr/bin/env bash
# Reset /dev/fb0 after the display container exits.
# Blanks the framebuffer (black screen) then tries to restore the Linux
# console by switching virtual terminals, which forces fbcon to redraw.
#
# Usage:
#   ./reset_display.sh            # uses /dev/fb0
#   ./reset_display.sh /dev/fb1   # alternate framebuffer

set -euo pipefail

FB="${1:-/dev/fb0}"

if [[ ! -e "$FB" ]]; then
    echo "Error: $FB not found" >&2
    exit 1
fi

echo "Blanking $FB..."
dd if=/dev/zero of="$FB" bs=4096 2>/dev/null || true

# Try to restore the Linux framebuffer console by switching VTs.
# This only works if fbcon is active (headless Pi with no desktop env).
if command -v chvt &>/dev/null && command -v fgconsole &>/dev/null; then
    cur=$(fgconsole 2>/dev/null || echo 1)
    alt=$(( cur == 1 ? 2 : 1 ))
    chvt "$alt" 2>/dev/null && chvt "$cur" 2>/dev/null && \
        echo "Console restored (VT$cur)" || \
        echo "chvt failed — display is blank"
else
    echo "chvt not available — display is blank"
fi
