# System Requirements for HADES-Lab

## Operating System
- Ubuntu 22.04+ or Debian 11+
- Linux kernel 5.15+ (for CUDA support)

## Hardware Requirements

### Minimum
- CPU: 8+ cores
- RAM: 32GB
- GPU: NVIDIA GPU with 12GB+ VRAM (RTX 3060 or better)
- Storage: 100GB SSD + 4TB HDD for data

### Recommended (Production)
- CPU: AMD Threadripper or Intel Xeon (16+ cores)
- RAM: 128GB+ ECC
- GPU: 2× NVIDIA RTX A6000 (48GB VRAM each) or similar
- Storage: 1TB NVMe for processing + 10TB+ for data

## Software Dependencies

### Python
- Python 3.11+ (required)
- Poetry for dependency management

### PHP (Required for ArangoDB Bridge)
- PHP 8.3+ CLI
- PHP Extensions:
  - php8.3-cli
  - php8.3-curl
  - php8.3-mbstring
  - php8.3-zip
- Composer (PHP package manager)
- ArangoDB PHP driver (`triagens/arangodb`)

### Database
- ArangoDB 3.11+ (Graph database)
- Configured for Unix socket at `/tmp/arangodb.sock`

### CUDA
- NVIDIA Driver 525+
- CUDA Toolkit 12.0+
- cuDNN 8.9+

## Installation Commands

### Ubuntu/Debian
```bash
# System packages
sudo apt update
sudo apt install -y git curl wget build-essential

# Python
sudo apt install -y python3.11 python3.11-dev python3-pip

# PHP and Composer
sudo apt install -y php8.3-cli php8.3-curl php8.3-mbstring php8.3-zip
curl -sS https://getcomposer.org/installer | php
sudo mv composer.phar /usr/local/bin/composer

# Poetry
curl -sSL https://install.python-poetry.org | python3 -

# NVIDIA drivers (if not installed)
sudo apt install -y nvidia-driver-525
```

### Python Dependencies
Managed via Poetry (`pyproject.toml`):
- torch 2.0+
- transformers
- sentence-transformers
- python-arango
- numpy, pandas
- And more...

### PHP Dependencies
Managed via Composer (`composer.json`):
- triagens/arangodb ^3.8

## Environment Variables

Required in `.env`:
```
ARANGO_PASSWORD=your_password
CUDA_VISIBLE_DEVICES=0,1
```

## Quick Setup

1. **Clone repository**:
   ```bash
   git clone https://github.com/yourusername/HADES-Lab.git
   cd HADES-Lab
   ```

2. **Run setup script**:
   ```bash
   ./setup/setup_local.sh
   ```

3. **Install Python dependencies**:
   ```bash
   poetry install
   poetry shell
   ```

4. **Install PHP dependencies**:
   ```bash
   composer require triagens/arangodb
   ```

5. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

6. **Verify installation**:
   ```bash
   python setup/verify_environment.py
   php core/database/arango/php_unix_bridge.php test
   ```

## Notes

- PHP is required because python-arango cannot use Unix sockets
- The PHP bridge (`core/database/arango/php_unix_bridge.php`) handles all database operations
- See `setup/php_arango_setup.md` for detailed PHP setup instructions