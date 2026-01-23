#!/bin/bash

# HADES-Lab Local Setup Script
# =============================
# Sets up the local development environment for HADES-Lab
# No PostgreSQL required - uses direct PDF processing with ArangoDB

set -e  # Exit on error

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
echo "  HADES-Lab Local Setup"
echo "  Direct PDF Processing with ArangoDB"
echo "============================================"
echo

# Check Python version
print_info "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.11"

if python3 - <<'PY' "$python_version" "$required_version"; then
import sys
inst = tuple(map(int, sys.argv[1].split('.')[:2]))
req = tuple(map(int, sys.argv[2].split('.')[:2]))
sys.exit(0 if inst >= req else 1)
PY
    print_success "Python $python_version found (>= $required_version required)"
else
    print_error "Python $python_version found (>= $required_version required)"
    exit 1
fi

# Check PHP version (required for ArangoDB bridge)
print_info "Checking PHP version..."
if command -v php &> /dev/null; then
    php_version=$(php --version 2>&1 | grep -Po '(?<=PHP )\d+\.\d+' | head -1)
    print_success "PHP $php_version found"

    # Check if Composer is installed
    if command -v composer &> /dev/null; then
        print_success "Composer is installed"

        # Check if ArangoDB PHP driver is installed
        if [ -f "composer.json" ] && grep -q "triagens/arangodb" composer.json; then
            print_success "ArangoDB PHP driver is installed"
        else
            print_warning "ArangoDB PHP driver not installed"
            echo "  Run: composer require triagens/arangodb"
        fi
    else
        print_warning "Composer not installed"
        echo "  Install with: apt install composer or see setup/php_arango_setup.md"
    fi
else
    print_warning "PHP not installed"
    echo "  PHP is required for ArangoDB Unix socket connections"
    echo "  Install with: apt install php8.3-cli php8.3-curl php8.3-mbstring php8.3-zip"
    echo "  See setup/php_arango_setup.md for full instructions"
fi

# Check for Poetry
print_info "Checking for Poetry..."
if command -v poetry &> /dev/null; then
    poetry_version=$(poetry --version | grep -Po '\d+\.\d+\.\d+')
    print_success "Poetry $poetry_version found"
else
    print_warning "Poetry not found. Installing..."
    installer="$(mktemp)"
    curl -sSL https://install.python-poetry.org -o "$installer"
    python3 "$installer"
    rm -f "$installer"
    export PATH="$HOME/.local/bin:$PATH"
    print_success "Poetry installed"
fi

# Get project root directory
_src="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
    resolved="$(readlink -f "${_src}" 2>/dev/null || true)"
    if [ -n "$resolved" ]; then
        _src="$resolved"
    fi
fi
SCRIPT_DIR="$(cd "$(dirname "${_src}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
if ! cd "$PROJECT_ROOT"; then
    print_error "Failed to enter project root: $PROJECT_ROOT"
    exit 1
fi

# Install dependencies
print_info "Installing Python dependencies with Poetry..."
poetry install
print_success "Dependencies installed"

# Check for GPU
print_info "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ' | sed 's/,$//')
    print_success "Found $gpu_count GPU(s): $gpu_names"
    
    # Check CUDA
    if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        print_success "CUDA is available for PyTorch"
    else
        print_warning "CUDA not available for PyTorch - GPU acceleration disabled"
    fi
else
    print_warning "No NVIDIA GPU detected - will run in CPU mode"
fi

# Check for ArangoDB connection
print_info "Checking ArangoDB connection..."
if [ -z "$ARANGO_PASSWORD" ]; then
    print_warning "ARANGO_PASSWORD not set"
    echo "Please set: export ARANGO_PASSWORD='your_password'"
else
    # Try to connect to ArangoDB
    if [ -f .env ]; then
        set -a
        . ./.env
        set +a
    fi
    arango_host=${ARANGO_HOST:-127.0.0.1}
    if curl -s -u root:"$ARANGO_PASSWORD" "http://$arango_host:8529/_api/version" > /dev/null 2>&1; then
        print_success "Connected to ArangoDB at $arango_host:8529"
    else
        print_warning "Could not connect to ArangoDB at $arango_host:8529"
        echo "Please ensure ArangoDB is running and ARANGO_PASSWORD is correct"
    fi
fi

# Check for data directory
print_info "Checking for ArXiv data directory..."
data_dir="/bulk-store/arxiv-data/pdf"
if [ -d "$data_dir" ]; then
    pdf_count=$(find "$data_dir" -name "*.pdf" 2>/dev/null | head -100 | wc -l)
    print_success "Found ArXiv data directory with PDFs"
    echo "  Sample count (first 100): $pdf_count PDFs"
else
    print_warning "ArXiv data directory not found at $data_dir"
    echo "  PDFs should be organized as: $data_dir/YYMM/arxiv_id.pdf"
fi

# Check RamFS for staging
print_info "Checking RamFS for staging directory..."
if mountpoint -q /dev/shm; then
    shm_size=$(df -h /dev/shm | awk 'NR==2 {print $2}')
    print_success "RamFS available at /dev/shm (Size: $shm_size)"
    
    # Create staging directory
    mkdir -p /dev/shm/acid_staging
    print_success "Staging directory ready at /dev/shm/acid_staging"
else
    print_warning "RamFS not available at /dev/shm"
    echo "  Pipeline will use regular filesystem for staging (slower)"
fi

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p logs
mkdir -p checkpoints
mkdir -p experiments/datasets
print_success "Directories created"

# Setup environment file
print_info "Creating environment template..."
cat > .env.template << 'EOF'
# ArangoDB Configuration
ARANGO_PASSWORD=your_password_here
ARANGO_HOST=localhost

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1  # Adjust based on your GPU setup
USE_GPU=true

# Optional: Paths
ARXIV_DATA_PATH=/bulk-store/arxiv-data
STAGING_PATH=/dev/shm/acid_staging
EOF

if [ ! -f .env ]; then
    cp .env.template .env
    print_success "Created .env file (please update with your credentials)"
else
    print_info ".env file already exists"
fi

# Summary
echo
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo
echo "Next steps:"
echo "1. Set your ArangoDB password:"
echo "   export ARANGO_PASSWORD='your_password'"
echo
echo "2. Activate the Poetry environment:"
echo "   poetry shell"
echo
echo "3. Verify the environment:"
echo "   python setup/verify_environment.py"
echo
echo "4. Verify database connection:"
echo "   python setup/verify_storage.py"
echo
print_success "Setup complete! Happy processing!"
