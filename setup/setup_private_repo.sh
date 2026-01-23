#!/bin/bash
# Setup script for private HADES repository

echo "=========================================="
echo "HADES Private Repository Setup"
echo "=========================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: ACID-compliant ArXiv pipeline achieving 6.8 papers/min

- Full ACID compliance verified
- Phase-separated architecture (extraction/embedding)
- Dual GPU processing with NVLink
- PostgreSQL + ArangoDB hybrid storage
- Jina v4 embeddings with late chunking
- Mathematical framework: C = (W·R·H)/T · Ctx^α"
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already initialized"
fi

echo ""
echo "Next steps:"
echo "1. Create a private repository on GitHub"
echo "2. Run: git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git"
echo "3. Run: git branch -M main"
echo "4. Run: git push -u origin main"
echo ""
echo "5. Add repository secrets in GitHub settings:"
echo "   - ARANGO_PASSWORD"
echo "   - PGPASSWORD"
echo ""
echo "6. Install dependencies locally:"
echo "   poetry install"
echo ""
echo "7. Set environment variables:"
echo "   export PGPASSWORD='your_password'"
echo "   export ARANGO_PASSWORD='your_password'"
echo "   export CUDA_VISIBLE_DEVICES=0,1"
echo ""
echo "Ready to begin experiments!"
echo "=========================================="