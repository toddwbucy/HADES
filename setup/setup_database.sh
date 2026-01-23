#!/bin/bash
# Setup ArXiv Repository Database with Production Best Practices
# ==============================================================
# This script sets up the arxiv_repository database with proper user management
# It integrates with the existing HADES-Lab environment setup

set -e  # Exit on error

# ==============================================================
# CONFIGURATION - Modify these values as needed
# ==============================================================

# Database Configuration
DB_NAME="${ARXIV_DB_NAME:-arxiv_repository}"

# Connection Configuration
# Prioritize Unix socket for local connections (much faster, more secure)
UNIX_SOCKET="${ARANGO_UNIX_SOCKET:-/run/arangodb3/arangodb.sock}"
USE_UNIX_SOCKET="${USE_UNIX_SOCKET:-true}"

# Fallback to HTTP if Unix socket not available
DB_HOST="${ARXIV_DB_HOST:-http://localhost:8529}"

# User Configuration
DB_ADMIN_USER="${ARXIV_ADMIN_USER:-arxiv_admin}"
DB_WRITER_USER="${ARXIV_WRITER_USER:-arxiv_writer}"
DB_READER_USER="${ARXIV_READER_USER:-arxiv_reader}"

# Password Configuration
# If not set, passwords will be auto-generated
DB_ADMIN_PASSWORD="${ARANGODB_PASSWORD:-}"
DB_WRITER_PASSWORD="${ARANGODB_PASSWORD:-}"
DB_READER_PASSWORD="${ARANGODB_PASSWORD:-}"

# Root password (required - from environment or .env)
ROOT_PASSWORD="${ARANGO_PASSWORD:-}"

# Collections to create
COLLECTIONS=(
    "arxiv_metadata"
    "arxiv_abstract_chunks"
    "arxiv_abstract_embeddings"
    "arxiv_structures"
    "arxiv_processing_order"
    "arxiv_processing_stats"
)

# Roles and Permissions
# rw = read-write, ro = read-only
ADMIN_PERMISSIONS="rw"
WRITER_PERMISSIONS="rw"
READER_PERMISSIONS="ro"

# Configuration Directory
CONFIG_DIR="${CONFIG_DIR:-config}"

# ==============================================================
# END CONFIGURATION
# ==============================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print header
echo "============================================"
echo "  ArXiv Repository Database Setup"
echo "  Production-grade with User Management"
echo "============================================"
echo

# Check for root password
if [ -z "$ROOT_PASSWORD" ]; then
    # Try to load from .env file
    if [ -f ".env" ]; then
        print_info "Loading credentials from .env file..."
        set -a
        # shellcheck disable=SC1091
        . ./.env
        set +a
        ROOT_PASSWORD="${ARANGO_PASSWORD:-}"
    fi

    # Check again after loading .env
    if [ -z "$ROOT_PASSWORD" ]; then
        print_error "ARANGO_PASSWORD not set"
        echo "Please either:"
        echo "  1. Export it: export ARANGO_PASSWORD='your_password'"
        echo "  2. Add it to .env file"
        exit 1
    fi
fi

print_success "Root password found"

# Check for Unix socket availability
if [ "$USE_UNIX_SOCKET" = "true" ] && [ -S "$UNIX_SOCKET" ]; then
    print_success "Unix socket found at $UNIX_SOCKET"
    print_info "Using Unix socket for optimal performance"
    CONNECTION_URL="http+unix://${UNIX_SOCKET//\//%2F}"
else
    if [ "$USE_UNIX_SOCKET" = "true" ]; then
        print_warning "Unix socket not found at $UNIX_SOCKET"
        print_info "Falling back to HTTP connection at $DB_HOST"
    else
        print_info "Using HTTP connection at $DB_HOST"
    fi
    CONNECTION_URL="$DB_HOST"
fi

print_info "Database name: $DB_NAME"
print_info "Admin user: $DB_ADMIN_USER"
print_info "Writer user: $DB_WRITER_USER"
print_info "Reader user: $DB_READER_USER"

# Parse command line arguments
DROP_EXISTING=false
AUTO_MODE=false
QUIET_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --drop-existing)
            DROP_EXISTING=true
            shift
            ;;
        --auto)
            AUTO_MODE=true
            shift
            ;;
        --quiet)
            QUIET_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --drop-existing  Drop and recreate database if it exists"
            echo "  --auto          Run in automatic mode (no prompts)"
            echo "  --quiet         Minimal output"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check Python and dependencies
print_info "Checking Python environment..."
if ! python3 -c "import arango" 2>/dev/null; then
    print_warning "python-arango not installed"
    print_info "Installing with pip..."
    pip install python-arango
fi

# Generate passwords if not provided
if [ -z "$DB_ADMIN_PASSWORD" ]; then
    DB_ADMIN_PASSWORD=$(openssl rand -base64 24)
    print_info "Generated admin password"
fi
if [ -z "$DB_WRITER_PASSWORD" ]; then
    DB_WRITER_PASSWORD=$(openssl rand -base64 24)
    print_info "Generated writer password"
fi
if [ -z "$DB_READER_PASSWORD" ]; then
    DB_READER_PASSWORD=$(openssl rand -base64 24)
    print_info "Generated reader password"
fi

# Set up database
print_info "Setting up $DB_NAME database..."

# Export configuration for Python script
export DB_NAME
export DB_HOST
export CONNECTION_URL
export UNIX_SOCKET
export USE_UNIX_SOCKET
export ROOT_PASSWORD
export DB_ADMIN_USER
export DB_ADMIN_PASSWORD
export DB_WRITER_USER
export DB_WRITER_PASSWORD
export DB_READER_USER
export DB_READER_PASSWORD
export ADMIN_PERMISSIONS
export WRITER_PERMISSIONS
export READER_PERMISSIONS
export CONFIG_DIR

# Convert collections array to JSON for Python
COLLECTIONS_JSON=$(printf '%s\n' "${COLLECTIONS[@]}" | jq -R . | jq -s .)
export COLLECTIONS_JSON

# Prepare Python command arguments
PYTHON_ARGS=()
if [ "$DROP_EXISTING" = true ]; then
    PYTHON_ARGS+=("--drop-existing")
fi
if [ "$AUTO_MODE" = true ]; then
    PYTHON_ARGS+=("--auto")
fi
if [ "$QUIET_MODE" = true ]; then
    PYTHON_ARGS+=("--quiet")
fi

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Run inline Python setup script
python3 - "${PYTHON_ARGS[@]}" <<'SETUP_PY'
import os
import sys
import json

from arango import ArangoClient
from arango.exceptions import DatabaseCreateError, CollectionCreateError, UserCreateError

# Get configuration from environment
db_name = os.environ.get('DB_NAME', 'arxiv_repository')
root_password = os.environ.get('ROOT_PASSWORD', '')
connection_url = os.environ.get('CONNECTION_URL', 'http://localhost:8529')
collections_json = os.environ.get('COLLECTIONS_JSON', '[]')

admin_user = os.environ.get('DB_ADMIN_USER', 'arxiv_admin')
admin_password = os.environ.get('DB_ADMIN_PASSWORD', '')
writer_user = os.environ.get('DB_WRITER_USER', 'arxiv_writer')
writer_password = os.environ.get('DB_WRITER_PASSWORD', '')
reader_user = os.environ.get('DB_READER_USER', 'arxiv_reader')
reader_password = os.environ.get('DB_READER_PASSWORD', '')

config_dir = os.environ.get('CONFIG_DIR', 'config')

drop_existing = '--drop-existing' in sys.argv
quiet_mode = '--quiet' in sys.argv

def log(msg):
    if not quiet_mode:
        print(msg)

# Parse collections
collections = json.loads(collections_json)

# Connect to ArangoDB
log(f"Connecting to ArangoDB at {connection_url}...")
client = ArangoClient(hosts=connection_url)
sys_db = client.db('_system', username='root', password=root_password)

# Create or get database
if sys_db.has_database(db_name):
    if drop_existing:
        log(f"Dropping existing database: {db_name}")
        sys_db.delete_database(db_name)
        sys_db.create_database(db_name)
        log(f"Recreated database: {db_name}")
    else:
        log(f"Database {db_name} already exists")
else:
    sys_db.create_database(db_name)
    log(f"Created database: {db_name}")

# Connect to the database
db = client.db(db_name, username='root', password=root_password)

# Create collections
for coll_name in collections:
    if not db.has_collection(coll_name):
        db.create_collection(coll_name)
        log(f"Created collection: {coll_name}")
    else:
        log(f"Collection {coll_name} already exists")

# Create users
users = [
    (admin_user, admin_password, 'rw'),
    (writer_user, writer_password, 'rw'),
    (reader_user, reader_password, 'ro'),
]

for username, password, permission in users:
    try:
        sys_db.create_user(username=username, password=password)
        log(f"Created user: {username}")
    except UserCreateError:
        sys_db.update_user(username=username, password=password)
        log(f"Updated user: {username}")

    # Grant permissions
    sys_db.update_permission(username=username, permission=permission, database=db_name)
    log(f"Granted {permission} on {db_name} to {username}")

# Save credentials to config file with restrictive permissions
os.makedirs(config_dir, exist_ok=True)
creds_file = os.path.join(config_dir, f"{db_name}.env")
with open(creds_file, 'w') as f:
    f.write(f"export ARXIV_DB_NAME={db_name}\n")
    f.write(f"export ARXIV_ADMIN_USER={admin_user}\n")
    f.write(f"export ARXIV_ADMIN_PASSWORD={admin_password}\n")
    f.write(f"export ARXIV_WRITER_USER={writer_user}\n")
    f.write(f"export ARXIV_WRITER_PASSWORD={writer_password}\n")
    f.write(f"export ARXIV_READER_USER={reader_user}\n")
    f.write(f"export ARXIV_READER_PASSWORD={reader_password}\n")

# Restrict permissions so only owner can read/write credentials
os.chmod(creds_file, 0o600)

log(f"Saved credentials to {creds_file} (mode 0600)")
print("Database setup completed successfully")
SETUP_PY

if [ $? -eq 0 ]; then
    print_success "Database setup completed successfully"
else
    print_error "Database setup failed"
    exit 1
fi

# Load the generated credentials
if [ -f "$CONFIG_DIR/$DB_NAME.env" ]; then
    print_info "Loading generated credentials..."
    source "$CONFIG_DIR/$DB_NAME.env"

    # Update .env file if it exists
    if [ -f ".env" ]; then
        print_info "Updating .env file with new credentials..."

        # Backup existing .env
        cp .env .env.backup

        # Add or update database credentials
        if grep -q "ARXIV_DB_NAME" .env; then
            # Update existing entries
            sed -i "s/^ARXIV_DB_NAME=.*/ARXIV_DB_NAME=$ARXIV_DB_NAME/" .env
            sed -i "s/^ARXIV_WRITER_PASSWORD=.*/ARXIV_WRITER_PASSWORD=$ARXIV_WRITER_PASSWORD/" .env
            sed -i "s/^ARXIV_READER_PASSWORD=.*/ARXIV_READER_PASSWORD=$ARXIV_READER_PASSWORD/" .env
            sed -i "s/^ARXIV_ADMIN_PASSWORD=.*/ARXIV_ADMIN_PASSWORD=$ARXIV_ADMIN_PASSWORD/" .env
        else
            # Add new entries
            echo "" >> .env
            echo "# ArXiv Repository Database" >> .env
            echo "ARXIV_DB_NAME=$ARXIV_DB_NAME" >> .env
            echo "ARXIV_WRITER_PASSWORD=$ARXIV_WRITER_PASSWORD" >> .env
            echo "ARXIV_READER_PASSWORD=$ARXIV_READER_PASSWORD" >> .env
            echo "ARXIV_ADMIN_PASSWORD=$ARXIV_ADMIN_PASSWORD" >> .env
        fi

        print_success "Updated .env file with database credentials"
    fi
else
    print_warning "Could not load generated credentials"
fi

# Test connection with new credentials
print_info "Testing database connection..."

python3 <<'PY'
import os
from datetime import UTC, datetime

from core.database.database_factory import DatabaseFactory
from core.database.arango import MemoryServiceError

db_name = os.environ.get('DB_NAME', 'arxiv_repository')
writer_user = os.environ.get('DB_WRITER_USER', 'arxiv_writer')
writer_password = os.environ.get('DB_WRITER_PASSWORD') or os.environ.get('ARANGO_PASSWORD')

if not writer_password:
    print('✗ Connection failed: database password not available')
    raise SystemExit(1)

try:
    client = DatabaseFactory.get_arango_memory_service(
        database=db_name,
        username=writer_user,
        password=writer_password,
    )
except Exception as exc:
    print(f'✗ Connection failed: {exc}')
    raise SystemExit(1)

try:
    result = client.execute_query('RETURN 1')
    assert result == [1], 'Query test failed'

    collections = client.execute_query(
        """
        FOR c IN _collections
            FILTER !STARTS_WITH(c.name, '_')
            RETURN c.name
        """
    )

    print(f'✓ Successfully connected as {writer_user}')
    print(f'  Collections: {len(collections)}')

    try:
        client.execute_transaction(
            write=["arxiv_metadata"],
            action="""
                function (params) {
                    const db = require('@arangodb').db;
                    db.arxiv_metadata.insert({_key: params.key, test: true, ts: params.ts});
                    db.arxiv_metadata.remove(params.key);
                    return true;
                }
            """,
            params={
                "key": "_test_permission",
                "ts": datetime.now(UTC).isoformat(),
            },
            wait_for_sync=True,
        )
        print('  Write permissions: ✓ Verified')
    except MemoryServiceError:
        print('  Write permissions: Read-only')
except Exception as exc:
    print(f'✗ Connection failed during verification: {exc}')
    raise SystemExit(1)
finally:
    client.close()

print('\n📊 Performance Note:')
print('  HTTP/2 over Unix sockets delivers the best throughput for local workflows.')
print('  Configure LISTEN_SOCKET/UPSTREAM_SOCKET if using the bundled proxies.')
PY

if [ $? -eq 0 ]; then
    print_success "Database connection test passed"
else
    print_error "Database connection test failed"
    exit 1
fi

# Summary
echo
echo "============================================"
echo "  Database Setup Complete!"
echo "============================================"
echo
echo "Database: $DB_NAME"
echo "Host: $DB_HOST"
echo
echo "Users created:"
echo "  • $DB_ADMIN_USER  - Full admin access"
echo "  • $DB_WRITER_USER - Read/write for processing"
echo "  • $DB_READER_USER - Read-only for monitoring"
echo
echo "Collections created:"
for collection in "${COLLECTIONS[@]}"; do
    echo "  • $collection"
done
echo
echo "Credentials saved to:"
echo "  • $CONFIG_DIR/$DB_NAME.env"
if [ -f ".env" ]; then
    echo "  • .env (updated)"
fi
echo
echo "To use in workflows:"
echo "  source $CONFIG_DIR/$DB_NAME.env"
echo "  python -m core.workflows.workflow_arxiv_sorted \\"
echo "    --database $DB_NAME \\"
echo "    --username $DB_WRITER_USER"
echo
print_success "Database ready for production use!"
