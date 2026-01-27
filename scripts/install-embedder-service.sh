#!/bin/bash
# Install the HADES Embedding Service as a systemd service
#
# Usage:
#   sudo ./scripts/install-embedder-service.sh
#
# After installation:
#   sudo systemctl start hades-embedder
#   sudo systemctl status hades-embedder
#   journalctl -u hades-embedder -f

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root (use sudo)"
    exit 1
fi

echo "Installing HADES Embedding Service..."

# Find the poetry virtualenv path
VENV_PATH=$(sudo -u todd bash -c 'cd '"$PROJECT_DIR"' && poetry env info -p 2>/dev/null')
if [[ -z "$VENV_PATH" ]]; then
    echo "Error: Could not find poetry virtualenv. Run 'poetry install' first."
    exit 1
fi

echo "Using virtualenv: $VENV_PATH"

# Update the service file with the correct path
SERVICE_FILE="$PROJECT_DIR/systemd/hades-embedder.service"
if [[ ! -f "$SERVICE_FILE" ]]; then
    echo "Error: Service file not found: $SERVICE_FILE"
    exit 1
fi

# Create a temporary copy with updated paths
TMP_SERVICE="/tmp/hades-embedder.service"
sed "s|/home/todd/.cache/pypoetry/virtualenvs/hades-[^/]*/bin/python|$VENV_PATH/bin/python|g" \
    "$SERVICE_FILE" > "$TMP_SERVICE"

# Also update the WorkingDirectory to the actual project directory
sed -i "s|WorkingDirectory=.*|WorkingDirectory=$PROJECT_DIR|g" "$TMP_SERVICE"

# Copy to systemd directory
cp "$TMP_SERVICE" /etc/systemd/system/hades-embedder.service
rm "$TMP_SERVICE"

# Create the runtime directory
mkdir -p /run/hades
chown todd:todd /run/hades
chmod 755 /run/hades

# Reload systemd
systemctl daemon-reload

# Enable the service
systemctl enable hades-embedder

echo ""
echo "Installation complete!"
echo ""
echo "Commands:"
echo "  sudo systemctl start hades-embedder   # Start the service"
echo "  sudo systemctl stop hades-embedder    # Stop the service"
echo "  sudo systemctl restart hades-embedder # Restart the service"
echo "  sudo systemctl status hades-embedder  # Check status"
echo "  journalctl -u hades-embedder -f       # View logs"
echo ""
echo "Test with:"
echo "  curl --unix-socket /run/hades/embedder.sock http://localhost/health"
