#!/usr/bin/env bash
# autostart.sh — install or remove the birdvision systemd service
#
# Usage:
#   ./scripts/autostart.sh install    # create service, enable, and start it
#   ./scripts/autostart.sh uninstall  # stop, disable, and remove the service

set -euo pipefail

SERVICE_NAME="birdvision"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${REPO_DIR}/docker-compose.pi.yml"
RUN_USER="${SUDO_USER:-evan}"

usage() {
    echo "Usage: $0 [install|uninstall]"
    exit 1
}

require_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "Error: this script must be run as root (use sudo)."
        exit 1
    fi
}

do_install() {
    require_root
    echo "Installing ${SERVICE_NAME} systemd service..."

    cat > "${SERVICE_FILE}" << EOF
[Unit]
Description=BirdVision Pi Pipeline
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_DIR}
ExecStartPre=/usr/bin/docker compose -f ${COMPOSE_FILE} build
ExecStart=/usr/bin/docker compose -f ${COMPOSE_FILE} up
ExecStop=/usr/bin/docker compose -f ${COMPOSE_FILE} down
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable "${SERVICE_NAME}.service"
    systemctl start "${SERVICE_NAME}.service"

    echo "Done. Service is enabled and running."
    echo "  Status:  sudo systemctl status ${SERVICE_NAME}"
    echo "  Logs:    sudo journalctl -u ${SERVICE_NAME} -f"
}

do_uninstall() {
    require_root
    echo "Removing ${SERVICE_NAME} systemd service..."

    if systemctl is-active --quiet "${SERVICE_NAME}.service"; then
        systemctl stop "${SERVICE_NAME}.service"
    fi

    if systemctl is-enabled --quiet "${SERVICE_NAME}.service" 2>/dev/null; then
        systemctl disable "${SERVICE_NAME}.service"
    fi

    if [[ -f "${SERVICE_FILE}" ]]; then
        rm "${SERVICE_FILE}"
    fi

    systemctl daemon-reload
    echo "Done. Service removed."
}

case "${1:-}" in
    install)   do_install ;;
    uninstall) do_uninstall ;;
    *)         usage ;;
esac
