#!/bin/bash
# Install the HADES Embedding Service as a systemd service
#
# Usage:
#   sudo ./scripts/install-embedder-service.sh [--user <username>]
#
# Options:
#   --user <username>  User to run the service as (default: $SUDO_USER or todd)
#
# After installation:
#   sudo systemctl start hades-embedder
#   sudo systemctl status hades-embedder
#   journalctl -u hades-embedder -f

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
SERVICE_USER="${SUDO_USER:-todd}"
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)
            SERVICE_USER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root (use sudo)"
    exit 1
fi

# Validate user exists
if ! id "$SERVICE_USER" &>/dev/null; then
    echo "Error: User '$SERVICE_USER' does not exist"
    exit 1
fi

SERVICE_GROUP=$(id -gn "$SERVICE_USER")

echo "Installing HADES Embedding Service..."
echo "  Service user: $SERVICE_USER"
echo "  Service group: $SERVICE_GROUP"

# Find the poetry virtualenv path
VENV_PATH=$(sudo -u "$SERVICE_USER" bash -c 'cd '"$PROJECT_DIR"' && poetry env info -p 2>/dev/null')
if [[ -z "$VENV_PATH" ]]; then
    echo "Error: Could not find poetry virtualenv. Run 'poetry install' first."
    exit 1
fi

echo "  Virtualenv: $VENV_PATH"

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

# Update the WorkingDirectory to the actual project directory
sed -i "s|WorkingDirectory=.*|WorkingDirectory=$PROJECT_DIR|g" "$TMP_SERVICE"

# Update User and Group
sed -i "s|^User=.*|User=$SERVICE_USER|g" "$TMP_SERVICE"
sed -i "s|^Group=.*|Group=$SERVICE_GROUP|g" "$TMP_SERVICE"

# Copy to systemd directory
cp "$TMP_SERVICE" /etc/systemd/system/hades-embedder.service
rm "$TMP_SERVICE"

# Create the runtime directory
echo "Creating runtime directory: /run/hades"
mkdir -p /run/hades
chown "$SERVICE_USER:$SERVICE_GROUP" /run/hades
chmod 755 /run/hades

# Create HuggingFace cache directory (required for ProtectHome=read-only)
echo "Creating HuggingFace cache directory: /var/cache/hades/huggingface"
mkdir -p /var/cache/hades/huggingface
chown -R "$SERVICE_USER:$SERVICE_GROUP" /var/cache/hades
chmod -R 755 /var/cache/hades

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
echo ""
echo "Note: The first start will download the model (~8GB) to /var/cache/hades/huggingface"
