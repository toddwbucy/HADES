#!/bin/bash
# Setup Python environment for HADES-Lab with uv and Poetry

set -e

echo "================================================"
echo "Setting up Python Environment for HADES-Lab"
echo "================================================"

# Check Python installation
echo -e "\n1. Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python found: $PYTHON_VERSION"
else
    echo "✗ Python3 not found. Please install Python 3.11 or higher"
    exit 1
fi

# Install uv (fast Python package manager)
echo -e "\n2. Installing uv package manager..."
if command -v uv &> /dev/null; then
    echo "✓ uv is already installed: $(uv --version)"
else
    echo "Installing uv..."
    install_script="$(mktemp)"
    curl -LsSf https://astral.sh/uv/install.sh -o "$install_script"
    sh "$install_script"
    rm -f "$install_script"
    
    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Add to shell profile if not already there
    if ! grep -q "/.cargo/bin" ~/.bashrc; then
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
        echo "Added uv to PATH in ~/.bashrc"
    fi
    
    echo "✓ uv installed successfully"
fi

# Install Poetry using uv
echo -e "\n3. Installing Poetry..."
if command -v poetry &> /dev/null; then
    echo "✓ Poetry is already installed: $(poetry --version)"
else
    echo "Installing Poetry using uv..."
    uv tool install poetry
    
    # Add Poetry to PATH
    export PATH="$HOME/.local/bin:$PATH"
    
    # Add to shell profile if not already there
    if ! grep -q "/.local/bin" ~/.bashrc; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        echo "Added Poetry to PATH in ~/.bashrc"
    fi
    
    echo "✓ Poetry installed successfully"
fi

# Configure Poetry for this project
echo -e "\n4. Configuring Poetry for HADES-Lab..."
# Get the directory where this script is located, then go to parent (project root)
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
    echo "✗ Failed to enter project root: $PROJECT_ROOT" >&2
    exit 1
fi

# Set Poetry to create virtual environments in project directory
poetry config virtualenvs.in-project true

# Check if pyproject.toml exists
if [ -f "pyproject.toml" ]; then
    echo "✓ pyproject.toml found"
    
    # Install dependencies
    echo -e "\n5. Installing project dependencies with Poetry..."
    poetry install
    
    echo "✓ Dependencies installed"
    
    # Also ensure uv can manage packages
    echo -e "\nSetting up uv for package management..."
    if [ -f "requirements.txt" ]; then
        echo "Installing from requirements.txt with uv..."
        uv pip install -r requirements.txt
    fi
else
    echo "✗ pyproject.toml not found - this shouldn't happen!"
    exit 1
fi

# Create .env template if it doesn't exist
echo -e "\n6. Setting up environment variables..."
if [ ! -f ".env" ]; then
    cat > .env.template << 'EOF'
# Database Configuration
ARANGO_PASSWORD=your-arango-password
ARANGO_HOST=localhost
ARANGO_PORT=8529

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1
USE_GPU=true

# Paths
ARXIV_DATA_PATH=/bulk-store/arxiv-data
STAGING_PATH=/dev/shm/acid_staging

# Processing Configuration
BATCH_SIZE=32
NUM_WORKERS=8
EOF
    echo "✓ Created .env.template (copy to .env and fill in your values)"
else
    echo "✓ .env file already exists"
fi

# Create activation script
echo -e "\n7. Creating activation script..."
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activate HADES-Lab Python environment

# Ensure uv and poetry are in PATH
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Activate Poetry virtual environment
echo "Activating Poetry virtual environment..."
poetry shell
EOF

chmod +x activate_env.sh

echo "================================================"
echo "✓ Python environment setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Copy .env.template to .env and fill in your credentials:"
echo "   cp .env.template .env"
echo "   nano .env"
echo ""
echo "2. Activate the environment:"
echo "   source ~/.bashrc  # To update PATH"
echo "   ./activate_env.sh # To enter Poetry shell"
echo ""
echo "3. Verify installation:"
echo "   uv --version"
echo "   poetry --version"
echo "   poetry show      # List installed packages"
echo ""
echo "4. For MCP servers, you can now use uv to run Python scripts:"
echo "   uv run python script.py"
echo "   # or within Poetry shell:"
echo "   python script.py"
