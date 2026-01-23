# PHP ArangoDB Bridge Setup

## Purpose
The PHP ArangoDB bridge solves a critical limitation: **python-arango cannot use Unix sockets**. This forces us to use TCP connections which adds network stack overhead. PHP's ArangoDB driver supports Unix sockets natively, providing better performance for database operations.

## Installation Steps

### 1. Install PHP 8.3
```bash
sudo apt install -y php8.3-cli php8.3-curl php8.3-mbstring php8.3-zip php8.3-dev
```

### 2. Install Composer
```bash
cd /tmp
curl -sS https://getcomposer.org/installer -o composer-setup.php
php composer-setup.php
sudo mv composer.phar /usr/local/bin/composer
rm composer-setup.php

# Verify
composer --version
```

### 3. Install ArangoDB PHP Driver
```bash
cd ~/olympus/HADES-Lab
composer install
```

This installs the dependencies listed in `composer.json`, including `triagens/arangodb`:
- `composer.json` - PHP dependencies
- `composer.lock` - Locked versions
- `vendor/` - PHP libraries

## Files Created

### `/core/database/arango/php_unix_bridge.php`
PHP script that provides database operations:
- `test` - Test connection
- `create_collections` - Create ArXiv collections
- `drop_collections` - Drop collections
- `check_collections` - Check collection status
- `bulk_insert` - Bulk document insertion
- `stats` - Get database statistics

### `/core/database/arango/unix_client.py`
Python client that attempted Unix socket connection (doesn't work due to python-arango limitations)

## Usage

### Test Connection
```bash
php core/database/arango/php_unix_bridge.php test
```

### Check Collections
```bash
php core/database/arango/php_unix_bridge.php check_collections
```

### Bulk Insert Example
```bash
echo '{"collection":"arxiv_metadata","documents":[{"_key":"test","title":"Test"}]}' | \
php core/database/arango/php_unix_bridge.php bulk_insert
```

## Current Status

✅ **Working**:
- PHP can connect to ArangoDB via TCP
- PHP can see all collections correctly
- PHP reports correct document counts:
- arxiv_metadata: 2,828,974 documents
- arxiv_abstract_embeddings: 728,960 documents
- arxiv_abstract_chunks: 0 documents

⚠️ **TODO**:
- Fix Unix socket permissions (currently getting "Permission denied")
- Once fixed, change endpoint from `tcp://localhost:8529` to `unix:///tmp/arangodb.sock`

## Python Integration

Python workflows can call the PHP bridge for database operations:

```python
import subprocess
import json

def check_collections():
    result = subprocess.run(
        ['php', 'core/database/arango/php_unix_bridge.php', 'check_collections'],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)
```

## Why This Matters

1. **Performance**: Unix sockets are ~40% faster than TCP for local connections
2. **Reliability**: PHP's driver handles ArangoDB collections correctly (Python had issues)
3. **Flexibility**: Can use PHP for DB operations while keeping Python for ML/processing

## Next Steps

1. Fix Unix socket permissions issue
2. Create Python wrapper class for PHP bridge
3. Integrate into workflow_arxiv_sorted_simple.py for database operations
4. Benchmark performance improvement vs pure Python approach
